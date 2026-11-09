"""
Strategy: Randomized Greedy Set Cover with Local Search Refinement.

1. Precompute all C(19,4) = 3876 t-subsets and C(19,7) = 50388 k-blocks as bitmasks.
2. Precompute coverage matrix (block covers t-subset).
3. Run multiple greedy passes:
   - One deterministic greedy (always pick max gain)
   - Several randomized greedy passes (pick with prob proportional to gain^alpha)
   This explores diverse block combinations.
4. Keep the best (smallest) valid covering.
5. Local search: iteratively remove non-critical blocks, then try 1-for-1 swaps.
"""

import time
import numpy as np
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    # 1. Precompute all t-subsets and their bitmasks
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    
    t_masks = np.array([0] * num_t, dtype=np.int64)
    for i, sub in enumerate(t_subsets):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_masks[i] = mask
        
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    k_masks = np.array([0] * num_k, dtype=np.int64)
    for i, block in enumerate(k_blocks):
        mask = 0
        for p in block:
            mask |= (1 << p)
        k_masks[i] = mask
        
    # Precompute coverage matrix in chunks
    cover_list = []
    chunk_size = 1000
    for start in range(0, num_k, chunk_size):
        end = min(start + chunk_size, num_k)
        k_chunk = k_masks[start:end]
        check = (t_masks[None, :] & ~k_chunk[:, None]) == 0
        cover_list.append(check.astype(np.uint8))
    
    def greedy_pass(randomized=False, alpha=1.0, rng=None):
        """Run one greedy pass. Returns list of block indices selected."""
        uncovered = np.ones(num_t, dtype=np.uint8)
        selected = []
        
        while np.sum(uncovered) > 0:
            if time.time() > deadline - 10:
                break
            
            best_idx = -1
            best_gain = 0
            gains = np.zeros(num_k, dtype=np.int32)
            
            for chunk_idx, cover_chunk in enumerate(cover_list):
                gains[chunk_idx*chunk_size : chunk_idx*chunk_size + len(cover_chunk)] = cover_chunk.dot(uncovered)
            
            if randomized:
                # Soft-max sampling: probability proportional to gain^alpha
                # Add small epsilon to avoid zero probability
                gains_float = gains.astype(np.float64)
                gains_float[gains_float == 0] = 0
                # Use power to control exploration
                if alpha != 1.0:
                    gains_float = np.power(gains_float, alpha)
                total = gains_float.sum()
                if total > 0:
                    probs = gains_float / total
                    # Sample from distribution
                    best_idx = rng.choice(num_k, p=probs)
                    best_gain = int(gains[best_idx])
                else:
                    break
            else:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                best_gain = int(max_gain)
                best_idx = int(np.argmax(gains))
            
            if best_gain == 0:
                break
                
            selected.append(best_idx)
            
            # Update uncovered
            chunk_idx = best_idx // chunk_size
            local_idx = best_idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            uncovered[cover_row == 1] = 0
            
        return selected
    
    def local_search(indices_selected, time_budget=30):
        """Apply local search to reduce block count. Returns refined indices."""
        if not indices_selected:
            return indices_selected
        
        ls_start = time.time()
        ls_deadline = ls_start + time_budget
        
        # Initialize counts
        counts = np.zeros(num_t, dtype=np.int32)
        for idx in indices_selected:
            chunk_idx = idx // chunk_size
            local_idx = idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            counts += cover_row
        
        # Phase 1: Iterative removal of non-critical blocks
        changed = True
        while changed and time.time() < ls_deadline - 5:
            changed = False
            new_indices = []
            for idx in indices_selected:
                chunk_idx = idx // chunk_size
                local_idx = idx % chunk_size
                cover_row = cover_list[chunk_idx][local_idx]
                
                critical_mask = (cover_row == 1) & (counts == 1)
                if not np.any(critical_mask):
                    counts -= cover_row
                    changed = True
                else:
                    new_indices.append(idx)
            indices_selected = new_indices
        
        # Phase 2: Try 1-for-1 swaps
        if time.time() < ls_deadline - 3:
            selected_set = set(indices_selected)
            non_selected = [i for i in range(num_k) if i not in selected_set]
            
            # Try swaps for each selected block
            for remove_idx in list(indices_selected):
                if time.time() > ls_deadline - 3:
                    break
                
                chunk_idx_r = remove_idx // chunk_size
                local_idx_r = remove_idx % chunk_size
                cover_row_remove = cover_list[chunk_idx_r][local_idx_r]
                
                unique_mask = (cover_row_remove == 1) & (counts == 1)
                n_unique = np.sum(unique_mask)
                
                if n_unique == 0:
                    continue
                
                # Try a few random replacements
                sample_size = min(100, len(non_selected))
                if sample_size > 0:
                    candidates = random.sample(non_selected, sample_size)
                    for add_idx in candidates:
                        if time.time() > ls_deadline - 3:
                            break
                        
                        chunk_idx_a = add_idx // chunk_size
                        local_idx_a = add_idx % chunk_size
                        cover_row_add = cover_list[chunk_idx_a][local_idx_a]
                        
                        missing = unique_mask & (cover_row_add == 0)
                        if not np.any(missing):
                            # Check validity after swap
                            new_counts = counts - cover_row_remove + cover_row_add
                            if np.all(new_counts >= 1):
                                # Perform swap
                                counts = new_counts
                                for i, idx in enumerate(indices_selected):
                                    if idx == remove_idx:
                                        indices_selected[i] = add_idx
                                        break
                                break
        
        return indices_selected
    
    # Run multiple greedy passes
    best_solution = None
    best_len = float('inf')
    
    # Pass 1: Deterministic greedy
    if time.time() < deadline - 20:
        sol = greedy_pass(randomized=False)
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Passes 2+: Randomized greedy with different alpha values
    rng = random.Random(42)
    alphas = [0.5, 0.7, 1.0, 1.5, 2.0]
    for alpha in alphas:
        if time.time() > deadline - 15:
            break
        sol = greedy_pass(randomized=True, alpha=alpha, rng=rng)
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # If we didn't get any solution (shouldn't happen), fall back
    if best_solution is None:
        best_solution = greedy_pass(randomized=False)
        best_len = len(best_solution)
    
    # Apply local search to the best solution
    if time.time() < deadline - 5:
        refined = local_search(best_solution, time_budget=min(30, max(5, deadline - time.time() - 5)))
        if refined and len(refined) <= len(best_solution):
            best_solution = refined
            best_len = len(refined)
    
    # Convert to list of lists
    return [list(k_blocks[i]) for i in best_solution]
