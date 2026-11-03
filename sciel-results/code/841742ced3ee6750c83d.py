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
            
    # 4. Calculate Weights for Weighted Greedy
    # Frequency of each t-subset across all k-blocks
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)
    
    # 5. Greedy Set Cover with Weighted Gain and Local Search
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        
        # Greedy loop
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate weighted gains
            gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly to encourage diversity
            best_idx = rng.choice(candidates)
            
            selected_blocks.append(best_idx)
            
            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False
            
        # Check if we finished within time limit
        if time.time() > deadline - 5.0:
            break
            
        # Post-optimization: Remove redundant blocks
        if selected_blocks:
            # Build coverage count
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in selected_blocks:
                coverage_count[block_t_indices[bi]] += 1
                
            remaining = list(selected_blocks)
            changed = True
            rng.shuffle(remaining)
            
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
                        changed = True
                        break
                        
            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]
                
            # Local Search: Shake and Repair
            if time.time() < deadline - 10.0 and len(remaining) > 0:
                shake_size = max(1, len(remaining) // 10)
                shake_indices = rng.sample(range(len(remaining)), min(shake_size, len(remaining)))
                to_remove = set(remaining[i] for i in shake_indices)
                new_remaining = [b for b in remaining if b not in to_remove]
                
                # Re-calculate uncovered based on new_remaining
                new_uncovered = np.ones(n_t, dtype=bool)
                for bi in new_remaining:
                    new_uncovered[block_t_indices[bi]] = False
                    
                # Re-greedy to fill gaps
                while True:
                    if time.time() > deadline - 5.0:
                        break
                    if not new_uncovered.any():
                        break
                    
                    gains = (new_uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = rng.choice(candidates)
                    new_remaining.append(best_idx)
                    new_uncovered[block_t_indices[best_idx]] = False
                    
                # Re-remove redundants
                if new_remaining:
                    coverage_count_new = np.zeros(n_t, dtype=np.int32)
                    for bi in new_remaining:
                        coverage_count_new[block_t_indices[bi]] += 1
                        
                    changed = True
                    rng.shuffle(new_remaining)
                    while changed:
                        if time.time() > deadline - 2.0:
                            break
                        changed = False
                        for i in range(len(new_remaining)):
                            bi = new_remaining[i]
                            ts = block_t_indices[bi]
                            if np.all(coverage_count_new[ts] > 1):
                                coverage_count_new[ts] -= 1
                                new_remaining.pop(i)
                                changed = True
                                break
                                
                    if len(new_remaining) < best_size:
                        best_size = len(new_remaining)
                        best_solution = [k_subsets[i] for i in new_remaining]

    if best_solution is None:
        return []
        
    return [list(b) for b in best_solution]
