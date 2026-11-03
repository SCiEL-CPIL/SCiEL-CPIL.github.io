import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for safety
    
    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    # Filling the array
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
            
    # Precompute t_to_blocks: list of lists of block indices
    # t_to_blocks[i] contains all block indices that cover t_subsets[i]
    t_to_blocks = [[] for _ in range(n_t)]
    # Iterate over blocks and their covered t-subsets
    # This is O(N_k * C(k,t))
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)
            
    # Convert to numpy arrays for faster intersection? 
    # Lists are fine for small intersections, but numpy might be better for set ops.
    # Let's keep as lists for now, or convert to sets if needed.
    # For intersection, we can use set intersection.
    t_to_blocks_sets = [set(lst) for lst in t_to_blocks]

    # 4. Greedy Set Cover with Multiple Restarts
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        selected_set = set()
        
        # Greedy loop
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gains
            gains = uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly
            best_idx = rng.choice(candidates)
            
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False
            
        if time.time() > deadline - 5.0:
            break
            
        # Post-optimization: Local Search with Redundancy Removal and Swaps
        if selected_blocks:
            # Build coverage count
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in selected_blocks:
                coverage_count[block_t_indices[bi]] += 1
                
            remaining = list(selected_blocks)
            remaining_set = set(remaining)
            
            improved = True
            while improved:
                if time.time() > deadline - 2.0:
                    break
                improved = False
                
                # 1. Remove redundant blocks
                changed = True
                while changed:
                    if time.time() > deadline - 2.0:
                        break
                    changed = False
                    for i in range(len(remaining)):
                        bi = remaining[i]
                        ts = block_t_indices[bi]
                        if np.all(coverage_count[ts] > 1):
                            coverage_count[ts] -= 1
                            remaining.pop(i)
                            remaining_set.remove(bi)
                            changed = True
                            break # Restart scan
                
                if time.time() > deadline - 2.0:
                    break

                # 2. Try swaps
                # For each block in remaining, check if it can be swapped
                # We iterate through a copy of remaining to avoid index issues
                for i in range(len(remaining)):
                    if time.time() > deadline - 2.0:
                        break
                    
                    bi = remaining[i]
                    ts = block_t_indices[bi]
                    
                    # Find unique t-subsets covered by bi
                    unique_ts = ts[coverage_count[ts] == 1]
                    
                    if len(unique_ts) == 0:
                        # This block is redundant, should have been removed
                        # But maybe coverage changed? 
                        # If all coverage > 1, it's redundant.
                        # If some == 1, it's not redundant.
                        # If len(unique_ts) == 0, it means no t-subset is uniquely covered by bi.
                        # This implies all t-subsets in bi are covered by >= 2 blocks.
                        # So bi is redundant.
                        coverage_count[ts] -= 1
                        remaining.pop(i)
                        remaining_set.remove(bi)
                        improved = True
                        break # Restart outer loop
                    
                    # Find candidates B that cover all unique_ts
                    # Intersection of t_to_blocks_sets for each t in unique_ts
                    candidates = t_to_blocks_sets[unique_ts[0]]
                    for t_idx in unique_ts[1:]:
                        candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                    
                    # Filter out blocks already in solution
                    candidates.difference_update(remaining_set)
                    
                    if candidates:
                        # Pick a random candidate
                        new_bi = rng.choice(list(candidates))
                        
                        # Perform swap
                        # Remove bi
                        coverage_count[block_t_indices[bi]] -= 1
                        remaining_set.remove(bi)
                        remaining.pop(i)
                        
                        # Add new_bi
                        coverage_count[block_t_indices[new_bi]] += 1
                        remaining_set.add(new_bi)
                        remaining.append(new_bi)
                        
                        improved = True
                        break # Restart outer loop after a swap to re-evaluate redundancies
            
            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]
                
    if best_solution is None:
        return []
        
    return [list(b) for b in best_solution]
