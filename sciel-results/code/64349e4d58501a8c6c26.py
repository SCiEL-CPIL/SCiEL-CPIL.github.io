import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, precompute the list of t-subset indices it covers
    # This is expensive to compute for all 50388 blocks, but we only need it for the ones we pick?
    # No, in the greedy we construct blocks on the fly.
    # But in local search, we have specific blocks.
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper to compute t-subset indices for a block
    def get_block_t_indices(block):
        return [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]

    # Multiple runs with different random seeds
    for run_id in range(200):  # Max 200 runs, but limited by time
        if time.time() > deadline - 5.0:
            break
            
        random.seed(run_id * 12345 + 67890)
        
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Pick a random uncovered t-subset to seed the block
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = random.choice(uncovered_indices)
            seed_subset = all_t_subsets[seed_idx]
            block = list(seed_subset)
            
            while len(block) < k:
                # Find x that covers most uncovered t-subsets
                best_x = -1
                best_gain = -1
                tied_candidates = []
                
                block_set = set(block)
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                
                # Optimization: Precompute the sorted tuples for rest + x
                # But x varies.
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # We can optimize this by noting that if we add x,
                    # the new t-subsets are {x} U rest for rest in C(block, t-1).
                    for rest in t_minus_1_subsets:
                        T = tuple(sorted(rest + (x,)))
                        if T in t_sub_to_idx:
                            idx = t_sub_to_idx[T]
                            if uncovered[idx]:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        tied_candidates = [x]
                    elif gain == best_gain:
                        tied_candidates.append(x)
                        
                if best_x == -1:
                    break
                    
                # Random tie-breaking
                if len(tied_candidates) > 1:
                    best_x = random.choice(tied_candidates)
                    
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            for sub in itertools.combinations(block, t):
                if sub in t_sub_to_idx:
                    uncovered[t_sub_to_idx[sub]] = False
            
        # End while uncovered
        
        # Local Optimization: Remove redundant blocks with incremental counts
        if len(blocks) < best_count:
            # Prepare for local search
            n_blocks = len(blocks)
            block_t_indices = [get_block_t_indices(b) for b in blocks]
            coverage_count = np.zeros(num_t, dtype=np.int32)
            
            for idxs in block_t_indices:
                coverage_count[idxs] += 1
                
            # Iteratively remove redundant blocks
            changed = True
            while changed:
                changed = False
                if time.time() > deadline - 2.0:
                    break
                    
                # Find a removable block
                # A block is removable if all its t-subsets have coverage > 1
                # We check blocks in random order to avoid bias
                order = list(range(len(blocks)))
                random.shuffle(order)
                
                removable_idx = -1
                for i in order:
                    # Check if block i is removable
                    is_removable = True
                    for sub_idx in block_t_indices[i]:
                        if coverage_count[sub_idx] <= 1:
                            is_removable = False
                            break
                    if is_removable:
                        removable_idx = i
                        break
                        
                if removable_idx != -1:
                    # Remove block
                    idxs_to_remove = block_t_indices[removable_idx]
                    for sub_idx in idxs_to_remove:
                        coverage_count[sub_idx] -= 1
                    
                    blocks.pop(removable_idx)
                    block_t_indices.pop(removable_idx)
                    changed = True
                    
        # End redundancy check
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    # If no blocks found, fallback to simple greedy
    if best_blocks is None:
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        while uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = uncovered_indices[0]
            seed_subset = all_t_subsets[seed_idx]
            block = list(seed_subset)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in t_minus_1_subsets:
                        T = tuple(sorted(rest + (x,)))
                        if T in t_sub_to_idx:
                            idx = t_sub_to_idx[T]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                if sub in t_sub_to_idx:
                    uncovered[t_sub_to_idx[sub]] = False
                    
        best_blocks = blocks
        best_count = len(blocks)
        
    return best_blocks
