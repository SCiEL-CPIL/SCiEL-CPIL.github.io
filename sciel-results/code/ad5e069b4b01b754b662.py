import numpy as np
import time
import itertools
import random

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute combinations of (t-1) subsets for gain calculation
    # This might be overkill, calculating on the fly is fast enough for small k-t
    
    # Deadline: 110 seconds to be safe (limit is 120, kill is 150)
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_blocks = None
    best_count = float('inf')
    
    # We will run multiple randomized greedy attempts
    while time.time() < deadline:
        # Shuffle the order of subsets to randomize the "first uncovered" choice
        shuffled_subsets = all_t_subsets[:]
        random.shuffle(shuffled_subsets)
        
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = []
        
        for s in shuffled_subsets:
            idx_s = subset_to_idx[s]
            if not uncovered[idx_s]:
                continue
                
            # Build a block containing s
            block = list(s)
            
            # Greedily add points to the block
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_set = set(block)
                
                # Calculate gain for each candidate point
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Gain is the number of new t-subsets covered by adding x
                    # These are subsets formed by (t-1) elements from 'block' and x
                    gain = 0
                    # Iterate over all (t-1) subsets of the current block
                    # t=4, so we iterate over 3-subsets of block
                    for rest in itertools.combinations(block, t - 1):
                        # Form the t-subset
                        new_subset = tuple(sorted(list(rest) + [x]))
                        # Check if it's in the mapping and if it's uncovered
                        if new_subset in subset_to_idx:
                            idx = subset_to_idx[new_subset]
                            if uncovered[idx]:
                                gain += 1
                                
                    # Add small random noise to break ties
                    gain += random.random() * 0.1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                block.append(best_point)
            
            # Sort the block for consistency (though not strictly required if we use sets)
            block.sort()
            current_blocks.append(block)
            
            # Update uncovered status
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered[idx] = False
            
            # Optional: Early termination if all covered? 
            # Not easy to check without scanning, but we can check if uncovered.sum() == 0
            # This is O(N) per block, which is fine.
            if not uncovered.any():
                break
        
        # Check if this run is better
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
            
            # If we hit the theoretical lower bound or a very good number, we might want to stop early?
            # Lower bound is 131. If we find 131, we can stop.
            if best_count <= 131:
                break

    if best_blocks is None:
        # Fallback: return a valid covering (should not happen if loop runs at least once)
        # But to be safe, run one deterministic greedy
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = []
        for s in all_t_subsets:
            idx_s = subset_to_idx[s]
            if not uncovered[idx_s]:
                continue
            block = list(s)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_subset = tuple(sorted(list(rest) + [x]))
                        if new_subset in subset_to_idx:
                            idx = subset_to_idx[new_subset]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            block.sort()
            current_blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered[idx] = False
        best_blocks = current_blocks

    return best_blocks
