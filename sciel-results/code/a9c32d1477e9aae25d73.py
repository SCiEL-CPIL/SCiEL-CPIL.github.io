"""Covering design C(v,k,t) using Iterated Local Search with perturbation and repair."""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 118  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute for each 7-subset, the list of t-subset indices it covers
    # We'll compute this on demand with caching
    block_tsub_cache = {}
    
    def get_block_tsubs(block):
        """Return list of t-subset indices covered by this block (tuple)."""
        bt = tuple(sorted(block))
        if bt not in block_tsub_cache:
            indices = []
            for ts in itertools.combinations(bt, t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    indices.append(idx)
            block_tsub_cache[bt] = indices
        return block_tsub_cache[bt]
    
    def build_coverage(blocks):
        """Build coverage count array and per-block t-subset lists."""
        cov = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs_list = []
        for block in blocks:
            tsubs = get_block_tsubs(block)
            block_tsubs_list.append(tsubs)
            for idx in tsubs:
                cov[idx] += 1
        return cov, block_tsubs_list
    
    def is_valid(cov):
        return cov.min() >= 1
    
    def greedy_repair(uncovered_indices, max_blocks=None):
        """Add blocks to cover all uncovered t-subsets. Returns list of new blocks."""
        if not uncovered_indices:
            return []
        
        uncovered_set = set(uncovered_indices)
        new_blocks = []
        
        # Greedily pick blocks that cover most uncovered t-subsets
        # Candidate blocks: all 7-subsets containing at least one uncovered t-subset
        # For efficiency, just try random 7-subsets and the ones containing uncovered t-subsets
        
        remaining = list(uncovered_set)
        if max_blocks is None:
            max_blocks = len(remaining) + 10
        
        for _ in range(max_blocks):
            if not remaining:
                break
            
            # Find the uncovered t-subset with fewest covering options (most constrained)
            # Just pick randomly for speed
            target_idx = remaining[rng.randint(0, len(remaining) - 1)]
            target_tsub = all_tsubs[target_idx]
            
            # Generate candidate blocks that contain this t-subset
            remaining_points = [p for p in range(v) if p not in target_tsub]
            need_more = k - t
            if need_more <= 0:
                candidate = list(target_tsub)
            else:
                extra = rng.sample(remaining_points, need_more)
                candidate = list(target_tsub) + extra
            
            candidate.sort()
            cand_tsubs = get_block_tsubs(candidate)
            
            # Count how many currently uncovered t-subsets this covers
            gain = sum(1 for idx in cand_tsubs if idx in uncovered_set)
            
            if gain > 0:
                new_blocks.append(candidate)
                for idx in cand_tsubs:
                    uncovered_set.discard(idx)
                remaining = [x for x in remaining if x not in cand_tsubs or x in uncovered_set]
                remaining = list(uncovered_set)
        
        return new_blocks
    
    def local_search(blocks, deadline_check=None):
        """Run local search: try to delete blocks, then try swaps."""
        blocks = list(blocks)
        cov, block_tsubs_list = build_coverage(blocks)
        
        improved = True
        while improved:
            if deadline_check and time.time() > deadline_check:
                break
            improved = False
            n = len(blocks)
            
            # Phase 1: Try to delete blocks
            for i in range(n):
                if deadline_check and time.time() > deadline_check:
                    break
                # Check if block i is redundant
                redundant = True
                for idx in block_tsubs_list[i]:
                    if cov[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for idx in block_tsubs_list[i]:
                        cov[idx] -= 1
                    blocks.pop(i)
                    block_tsubs_list.pop(i)
                    improved = True
                    break
            
            # Phase 2: Try swaps - replace a block with a better one
            if not improved and n > 0:
                # Try swapping some blocks
                for i in range(n):
                    if deadline_check and time.time() > deadline_check:
                        break
                    # Try a few random replacements
                    for attempt in range(5):
                        # Generate a random 7-subset
                        new_block = sorted(rng.sample(range(v), k))
                        new_tsubs = get_block_tsubs(new_block)
                        
                        # Check: if we replace block i with new_block, is coverage still valid?
                        # Block i's t-subsets lose one cover, new_block's gain one
                        valid = True
                        for idx in block_tsubs_list[i]:
                            if cov[idx] <= 1 and idx not in set(new_tsubs):
                                valid = False
                                break
                        if valid:
                            # Apply swap
                            for idx in block_tsubs_list[i]:
                                cov[idx] -= 1
                            for idx in new_tsubs:
                                cov[idx] += 1
                            blocks[i] = new_block
                            block_tsubs_list[i] = new_tsubs
                            improved = True
                            break
                    if improved:
                        break
        
        return blocks
    
    # Initial solution: greedy construction
    def greedy_build():
        """Standard greedy covering."""
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        blocks = []
        
        while uncovered.any():
            # Find most constrained uncovered t-subset
            # Pick randomly among uncovered for diversity
            uncovered_indices = np.where(uncovered > 0)[0]
            target_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            target_tsub = all_tsubs[target_idx]
            
            # Build a block around this t-subset
            block = list(target_tsub)
            remaining_points = [p for p in range(v) if p not in block]
            need_more = k - t
            
            # Greedily add points that cover most uncovered t-subsets
            for _ in range(need_more):
                if not remaining_points:
                    break
                best_gain = -1
                best_point = remaining_points[0]
                for p in remaining_points:
                    # Count uncovered t-subsets that would be covered by adding p
                    gain = 0
                    for combo in itertools.combinations(block, t-1):
                        ts = tuple(sorted(combo + (p,)))
                        idx = tsub_index.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                    elif gain == best_gain and rng.random() < 0.3:
                        best_point = p
                block.append(best_point)
                remaining_points.remove(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update coverage
            for idx in get_block_tsubs(block):
                uncovered[idx] = 0
        
        return blocks
    
    # Get initial solution
    best_blocks = greedy_build()
    best_count = len(best_blocks)
    
    # ILS loop
    current_blocks = list(best_blocks)
    temperature = 0.5
    cooling = 0.995
    
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        
        # Perturbation: remove m random blocks and repair
        m = rng.randint(1, max(1, len(current_blocks) // 10))
        perturbed = list(current_blocks)
        
        # Remove m random blocks
        to_remove = rng.sample(range(len(perturbed)), min(m, len(perturbed)))
        to_remove.sort(reverse=True)
        for i in to_remove:
            perturbed.pop(i)
        
        # Check coverage and repair
        cov, _ = build_coverage(perturbed)
        if cov.min() < 1:
            uncovered_indices = np.where(cov < 1)[0].tolist()
            new_blocks = greedy_repair(uncovered_indices)
            perturbed.extend(new_blocks)
        
        # Local search
        perturbed = local_search(perturbed, deadline - 5)
        
        # Accept if better or with probability based on temperature
        if len(perturbed) < best_count:
            best_blocks = list(perturbed)
            best_count = len(perturbed)
            current_blocks = list(perturbed)
        elif len(perturbed) == best_count:
            # Keep for diversity
            if rng.random() < 0.5:
                current_blocks = list(perturbed)
        else:
            # Simulated annealing acceptance
            delta = len(perturbed) - best_count
            if rng.random() < np.exp(-delta / max(temperature, 0.01)):
                current_blocks = list(perturbed)
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.1  # Reheat occasionally
    
    # Final local search on best
    if time.time() < deadline - 3:
        best_blocks = local_search(best_blocks, deadline - 3)
    
    return best_blocks
