import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Allow 110 seconds for computation to be safe
    deadline = start_time + 110.0
    
    # Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    # Precompute block_indices: shape (num_blocks, C(k,t))
    # C(7,4) = 35
    block_indices = np.empty((num_blocks, 35), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_indices[i, j] = subset_to_idx[sub]
            
    # Map block tuple to its index in all_blocks for fast lookup in reduce
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    def greedy_once(seed):
        np.random.seed(seed)
        uncovered = np.ones(num_subsets, dtype=bool)
        valid = np.ones(num_blocks, dtype=bool)
        blocks_used = []
        
        while uncovered.any():
            # Calculate gains for all valid blocks
            gains = uncovered[block_indices].sum(axis=1)
            gains[~valid] = -1
            max_gain = np.max(gains)
            
            if max_gain <= 0:
                # Safety break, shouldn't happen for feasible problems
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = np.random.choice(candidates)
            
            blocks_used.append(all_blocks[best_idx])
            uncovered[block_indices[best_idx]] = False
            valid[best_idx] = False
            
        return blocks_used

    def reduce(blocks):
        # Count coverage of each subset
        covered_count = np.zeros(num_subsets, dtype=np.int32)
        
        for b in blocks:
            idx = block_to_idx[b]
            covered_count[block_indices[idx]] += 1
            
        new_blocks = []
        for b in blocks:
            idx = block_to_idx[b]
            # Check if any subset is covered only by this block
            can_remove = True
            for s_idx in block_indices[idx]:
                if covered_count[s_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block and update counts
                for s_idx in block_indices[idx]:
                    covered_count[s_idx] -= 1
            else:
                new_blocks.append(b)
                
        return new_blocks

    best_blocks = None
    seed = 0
    
    # Run multiple greedy instances
    while time.time() < deadline:
        blocks = greedy_once(seed)
        reduced_blocks = reduce(blocks)
        
        if best_blocks is None or len(reduced_blocks) < len(best_blocks):
            best_blocks = reduced_blocks
            
        seed += 1
        
    # If no blocks found (should not happen), return empty or fallback
    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]
