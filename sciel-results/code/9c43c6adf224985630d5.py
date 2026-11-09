"""Improved program for C(19,7,4) covering design using GRASP + local search."""
import itertools
import time
import random
from collections import defaultdict
import math


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point, the list of t-subsets containing it
    point_to_tsubs = defaultdict(list)
    for i, s in enumerate(all_t_subsets):
        for x in s:
            point_to_tsubs[x].append(i)
    
    def greedy_cover(rng, alpha=1.0, max_blocks=None):
        """
        Randomized greedy set cover.
        alpha: 1.0 = deterministic best, <1.0 = randomized (pick from top candidates)
        """
        covered = [False] * total_t_subsets
        uncovered_count = total_t_subsets
        blocks = []
        
        if max_blocks is None:
            max_blocks = 10000
        
        while uncovered_count > 0 and len(blocks) < max_blocks:
            if time.time() > deadline:
                break
            
            # For each candidate block (k-subset), compute gain = number of uncovered t-subsets it covers
            # But we can't enumerate all C(19,7)=50388 blocks each time... 
            # Instead, use a smarter approach: find the t-subset with fewest covering options,
            # then build a block around it.
            
            # Find an uncovered t-subset
            # For efficiency, track uncovered t-subsets
            uncovered_tsubs = [i for i in range(total_t_subsets) if not covered[i]]
            if not uncovered_tsubs:
                break
            
            # Pick a random uncovered t-subset (or the one with fewest options)
            # For simplicity, pick randomly from uncovered
            start_idx = uncovered_tsubs[rng.randrange(len(uncovered_tsubs))]
            start_subset = all_t_subsets[start_idx]
            
            # Build a block greedily: start with start_subset, add points that cover most uncovered t-subsets
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Count uncovered t-subsets that would be covered by adding x
                    # A t-subset is covered if it contains x and all other t-1 elements are in block
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(combo + (x,)))
                        if candidate in subset_to_idx:
                            idx = subset_to_idx[candidate]
                            if not covered[idx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(x)
                
                if best_gain <= 0 or not best_candidates:
                    # Pick a random point not in block
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = rng.choice(available)
                    else:
                        break
                else:
                    # Pick from best candidates (randomized if alpha < 1.0)
                    if alpha < 1.0 and len(best_candidates) > 1:
                        # Pick randomly from top candidates
                        best_point = rng.choice(best_candidates)
                    else:
                        best_point = best_candidates[0]
                
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) < k:
                # Fill with random points
                block_set = set(block)
                for x in range(v):
                    if x not in block_set and len(block) < k:
                        block.append(x)
                    if len(block) == k:
                        break
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            
            # Update covered array
            for t_subset in itertools.combinations(block, t):
                idx = subset_to_idx[t_subset]
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
        
        return blocks
    
    def local_search(blocks, rng):
        """
        Try to remove redundant blocks.
        A block is redundant if all its t-subsets are covered by other blocks.
        """
        if not blocks:
            return blocks
        
        # Count how many times each t-subset is covered
        tsub_counts = [0] * total_t_subsets
        for block in blocks:
            for t_subset in itertools.combinations(block, t):
                idx = subset_to_idx[t_subset]
                tsub_counts[idx] += 1
        
        # Try to remove blocks that don't cover any unique t-subset
        new_blocks = []
        for i, block in enumerate(blocks):
            # Check if this block covers any t-subset that no other block covers
            is_needed = False
            for t_subset in itertools.combinations(block, t):
                idx = subset_to_idx[t_subset]
                if tsub_counts[idx] == 1:
                    is_needed = True
                    break
            
            if is_needed:
                new_blocks.append(block)
            else:
                # Remove this block and update counts
                for t_subset in itertools.combinations(block, t):
                    idx = subset_to_idx[t_subset]
                    tsub_counts[idx] -= 1
        
        return new_blocks
    
    def swap_search(blocks, rng, time_budget=10.0):
        """
        Try to swap blocks to reduce count or maintain count with better structure.
        For now, just try removing one block and re-adding a better one.
        """
        if time.time() > start_time + time_budget:
            return blocks
        
        best_blocks = blocks
        best_len = len(blocks)
        
        # Try removing each block and see if we can cover the gap with one new block
        for i in range(len(blocks)):
            if time.time() > start_time + time_budget:
                break
            
            candidate_blocks = blocks[:i] + blocks[i+1:]
            
            # Find uncovered t-subsets after removal
            covered = [False] * total_t_subsets
            for block in candidate_blocks:
                for t_subset in itertools.combinations(block, t):
                    idx = subset_to_idx[t_subset]
                    covered[idx] = True
            
            uncovered = [i for i in range(total_t_subsets) if not covered[i]]
            if not uncovered:
                # We can remove this block!
                best_blocks = candidate_blocks
                best_len = len(candidate_blocks)
                break
            
            # Try to find a single block that covers all uncovered t-subsets
            # This is a hitting set problem, but for small uncovered sets, we can try
            if len(uncovered) <= 10:
                # Try to find a block that covers all uncovered
                for block_candidate in itertools.combinations(range(v), k):
                    covers_all = True
                    for uidx in uncovered:
                        if uidx not in [subset_to_idx[s] for s in itertools.combinations(block_candidate, t)]:
                            covers_all = False
                            break
                    if covers_all:
                        new_blocks = candidate_blocks + [list(block_candidate)]
                        if len(new_blocks) < best_len:
                            best_blocks = new_blocks
                            best_len = len(new_blocks)
                        break
        
        return best_blocks
    
    # Main search loop: multiple GRASP iterations
    best_solution = None
    best_len = float('inf')
    
    num_iterations = 0
    while time.time() < deadline - 5:
        num_iterations += 1
        
        # Vary alpha for diversity
        alpha = 1.0 if num_iterations % 3 == 0 else 0.7 + 0.3 * rng.random()
        
        # Greedy construction
        blocks = greedy_cover(rng, alpha)
        
        if not blocks:
            continue
        
        # Local search: remove redundant blocks
        blocks = local_search(blocks, rng)
        
        # Check if valid
        covered = [False] * total_t_subsets
        for block in blocks:
            for t_subset in itertools.combinations(block, t):
                idx = subset_to_idx[t_subset]
                covered[idx] = True
        
        if all(covered):
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_solution = blocks
                # Try swap search
                if time.time() < deadline - 15:
                    best_solution = swap_search(best_solution, rng, time_budget=min(10, deadline - time.time() - 5))
        
        # If we've found a good solution, do more local search
        if best_solution and len(best_solution) <= 160:
            # Try more aggressive local search
            pass
    
    # Return the best solution found
    if best_solution is None:
        # Fallback: deterministic greedy
        blocks = greedy_cover(rng, alpha=1.0)
        blocks = local_search(blocks, rng)
        return blocks
    
    return best_solution
