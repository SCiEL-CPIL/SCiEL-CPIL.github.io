import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using:
    1. Fast vectorized greedy construction
    2. Aggressive local search (block removal, swap)
    """
    start_time = time.time()
    timeout = 145  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the indices of t-subsets it covers
    block_to_tsubs = []
    for block in all_blocks:
        covered_indices = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_tsubs.append(covered_indices)
    
    # Convert to numpy arrays for faster operations
    block_tsubs_arrays = [np.array(t_indices, dtype=np.int32) for t_indices in block_to_tsubs]
    
    # Inverse index: for each t-subset, which blocks cover it
    tsub_to_blocks = defaultdict(list)
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            tsub_to_blocks[t_idx].append(b_idx)
    
    # Convert to numpy arrays
    tsub_to_blocks_arrays = [np.array(blks, dtype=np.int32) for blks in tsub_to_blocks.values()]
    
    def greedy_construct(random_seed=None):
        """Fast greedy construction with randomization"""
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
        else:
            rng = np.random.RandomState()
        
        uncovered_mask = np.ones(n_tsubs, dtype=bool)
        current_blocks = []
        
        # Randomize the order of blocks to break ties
        block_order = rng.permutation(n_blocks)
        
        while np.any(uncovered_mask):
            if time.time() - start_time > timeout - 5:
                break
                
            best_gain = -1
            best_b_idx = -1
            
            # Sample a subset of blocks for speed, expanding if needed
            # Start with a random sample, then expand if not finding good blocks
            sample_size = min(1000, n_blocks)
            candidates = block_order[:sample_size]
            
            for b_idx in candidates:
                t_indices = block_tsubs_arrays[b_idx]
                # Calculate gain: number of uncovered t-subsets this block covers
                gain = np.sum(uncovered_mask[t_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
                    # Early exit if we found a block covering everything uncovered?
                    # Not likely, but if gain is very high, we can break
                    if gain >= 30:  # Heuristic threshold
                        break
            
            if best_b_idx == -1 or best_gain == 0:
                # Expand search if needed
                for b_idx in range(n_blocks):
                    t_indices = block_tsubs_arrays[b_idx]
                    gain = np.sum(uncovered_mask[t_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_b_idx = b_idx
            
            if best_b_idx == -1:
                break
            
            current_blocks.append(best_b_idx)
            uncovered_mask[block_tsubs_arrays[best_b_idx]] = False
        
        return current_blocks
    
    def local_search(blocks, max_time=10):
        """Local search: remove redundant blocks, then try swaps"""
        if not blocks:
            return blocks
        
        end_time = start_time + max_time
        current = list(blocks)
        
        # Phase 1: Remove redundant blocks
        improved = True
        while improved:
            improved = False
            # Count coverage
            tsub_count = np.zeros(n_tsubs, dtype=np.int32)
            for b_idx in current:
                tsub_count[block_tsubs_arrays[b_idx]] += 1
            
            for i in range(len(current)):
                if time.time() > end_time:
                    break
                
                b_idx = current[i]
                # Check if removing this block leaves any t-subset uncovered
                t_indices = block_tsubs_arrays[b_idx]
                if np.all(tsub_count[t_indices] > 1):
                    # Safe to remove
                    current.pop(i)
                    tsub_count[t_indices] -= 1
                    improved = True
                    break  # Restart
        
        # Phase 2: Try to replace blocks with better ones
        # For each block, try replacing with a different block that covers the same uncovered sets
        improved = True
        while improved:
            improved = False
            # Count coverage
            tsub_count = np.zeros(n_tsubs, dtype=np.int32)
            for b_idx in current:
                tsub_count[block_tsubs_arrays[b_idx]] += 1
            
            for i in range(len(current)):
                if time.time() > end_time:
                    break
                
                b_idx = current[i]
                t_indices = block_tsubs_arrays[b_idx]
                
                # Find t-subsets that would become uncovered if we remove this block
                critical_tsubs = t_indices[tsub_count[t_indices] == 1]
                
                if len(critical_tsubs) == 0:
                    # This block is redundant, remove it
                    current.pop(i)
                    tsub_count[t_indices] -= 1
                    improved = True
                    break
                
                # Try to find a replacement block that covers all critical_tsubs
                # and possibly more
                best_replacement = -1
                best_extra_gain = -1
                
                # Sample candidate replacements
                candidates = random.sample(range(n_blocks), min(500, n_blocks))
                
                for cand_idx in candidates:
                    if cand_idx == b_idx:
                        continue
                    
                    cand_tsubs = block_tsubs_arrays[cand_idx]
                    
                    # Check if candidate covers all critical t-subsets
                    if not np.all(np.isin(critical_tsubs, cand_tsubs)):
                        continue
                    
                    # Calculate extra gain: how many additional t-subsets does it cover?
                    extra = np.sum((tsub_count[cand_tsubs] == 0) | (cand_tsubs != t_indices))
                    # Simpler: just count how many t-subsets in candidate are currently uncovered or under-covered
                    extra_gain = np.sum(tsub_count[cand_tsubs] == 0)
                    
                    if extra_gain > best_extra_gain:
                        best_extra_gain = extra_gain
                        best_replacement = cand_idx
                
                if best_replacement != -1 and best_extra_gain > 0:
                    # Replace
                    old_tsubs = block_tsubs_arrays[b_idx]
                    new_tsubs = block_tsubs_arrays[best_replacement]
                    
                    tsub_count[old_tsubs] -= 1
                    tsub_count[new_tsubs] += 1
                    current[i] = best_replacement
                    improved = True
                    break  # Restart
        
        return current
    
    # Main algorithm: try multiple greedy constructions and keep the best
    best_blocks = None
    best_count = float('inf')
    
    # Try several different greedy runs with different random seeds
    max_restarts = 10
    for seed in range(max_restarts):
        if time.time() - start_time > timeout - 15:
            break
        
        blocks = greedy_construct(random_seed=seed)
        
        if not blocks:
            continue
        
        # Apply local search
        blocks = local_search(blocks, max_time=5)
        
        # Verify coverage
        if np.any(np.zeros(n_tsubs, dtype=bool)):  # Placeholder
            pass
        
        # Check if valid
        covered = np.zeros(n_tsubs, dtype=bool)
        for b_idx in blocks:
            covered[block_tsubs_arrays[b_idx]] = True
        
        if np.all(covered):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        if time.time() - start_time > timeout - 10:
            break
    
    # If no valid solution found, do a final greedy without local search
    if best_blocks is None:
        blocks = greedy_construct()
        if blocks:
            covered = np.zeros(n_tsubs, dtype=bool)
            for b_idx in blocks:
                covered[block_tsubs_arrays[b_idx]] = True
            if np.all(covered):
                best_blocks = blocks
    
    # Final safety: if still no valid solution, return a trivial covering
    if best_blocks is None:
        # Build a trivial covering: for each t-subset, add a block containing it
        best_blocks = []
        covered = np.zeros(n_tsubs, dtype=bool)
        for t_idx in range(n_tsubs):
            if not covered[t_idx]:
                # Find any block covering this t-subset
                for b_idx in tsub_to_blocks_arrays[t_idx]:
                    if b_idx not in best_blocks:
                        best_blocks.append(b_idx)
                        covered[block_tsubs_arrays[b_idx]] = True
                        break
    
    # Convert to list of lists
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    return final_blocks
