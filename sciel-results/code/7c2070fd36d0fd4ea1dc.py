import numpy as np
import time
import itertools

def construct(q, n, radius):
    return _construct_impl(q, n, radius)

def _construct_impl(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_space = q ** n
    if total_space > 1000000:
        return []

    # Generate all possible centers
    all_centers = [list(c) for c in itertools.product(range(q), repeat=n)]
    center_to_idx = {tuple(c): i for i, c in enumerate(all_centers)}
    
    # Build Incidence Matrix A: N x M (Points x Centers)
    # A[i, j] = 1 if center j covers point i
    A = np.zeros((total_space, total_space), dtype=bool)
    
    for j, center in enumerate(all_centers):
        A[center_to_idx[tuple(center)], j] = True
        if radius >= 1:
            for pos in range(n):
                orig_val = center[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = center.copy()
                    new_word[pos] = new_val
                    idx = center_to_idx[tuple(new_word)]
                    A[idx, j] = True

    def greedy_vectorized(seed):
        rng = np.random.default_rng(seed)
        covered = np.zeros(total_space, dtype=bool)
        chosen = []
        
        while True:
            if time.time() > deadline - 5.0:
                return chosen
            
            uncovered = ~covered
            if not uncovered.any():
                break
                
            # Compute gains for all centers
            # gains[j] = sum(uncovered[i] for i where A[i, j] is True)
            # A is N x M. A.T is M x N.
            # gains = A.T @ uncovered
            gains = A.T @ uncovered
            
            max_gain = gains.max()
            if max_gain == 0:
                break
                
            # Random tie-breaking
            candidates = np.where(gains == max_gain)[0]
            j = candidates[rng.integers(len(candidates))]
            
            chosen.append(j)
            covered[A[:, j]] = True
            
        return chosen

    def local_search_vectorized(chosen):
        if not chosen:
            return chosen
            
        chosen_set = set(chosen)
        # K[i] = number of chosen centers covering point i
        # K = A @ I_chosen
        I_chosen = np.zeros(total_space, dtype=bool)
        for j in chosen:
            I_chosen[j] = True
        K = A @ I_chosen # N x 1 vector
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try to remove redundant centers
            # A center j is redundant if for all i where A[i, j] is True, K[i] > 1
            # i.e., no unique points.
            
            # We can check this for all j in chosen_set
            # But iterating over chosen_set in Python is fine if size is ~500.
            
            for j in list(chosen_set):
                # Unique points for j: i such that A[i, j] and K[i] == 1
                # mask_j = A[:, j]
                # unique_mask = mask_j & (K == 1)
                # if not unique_mask.any(): redundant
                
                mask_j = A[:, j]
                unique_mask = mask_j & (K == 1)
                
                if not unique_mask.any():
                    # Remove j
                    chosen_set.remove(j)
                    I_chosen[j] = False
                    K -= A[:, j] # Subtract coverage of j
                    improved = True
                    break # Restart loop
                    
            if improved:
                continue
                
            # Try swaps
            # For each j in chosen_set, find unique points U_j
            # Find k not in chosen_set such that k covers U_j
            
            # To speed up, we can prioritize j with fewest unique points?
            # Or just iterate.
            
            for j in list(chosen_set):
                mask_j = A[:, j]
                unique_mask = mask_j & (K == 1)
                unique_indices = np.where(unique_mask)[0]
                
                if len(unique_indices) == 0:
                    # Should have been caught above, but just in case
                    chosen_set.remove(j)
                    I_chosen[j] = False
                    K -= A[:, j]
                    improved = True
                    break
                    
                # Find k not in chosen_set covering all unique_indices
                # Candidates are all k not in chosen_set
                # We can check all k at once?
                # A[unique_indices, :] is a boolean matrix of shape (len(unique_indices), M)
                # We want columns k where all entries are True.
                # And k not in chosen_set.
                
                sub_A = A[unique_indices, :]
                # all_covered = np.all(sub_A, axis=0) # M x 1 bool
                # valid_k = np.where(all_covered)[0]
                # valid_k = [k for k in valid_k if k not in chosen_set]
                
                # This creates a large array if unique_indices is large.
                # But unique_indices is usually small (ball size is 17, unique points <= 17).
                # So sub_A is (<=17, 6561).
                # np.all(sub_A, axis=0) is fast.
                
                all_covered = np.all(sub_A, axis=0)
                potential_k = np.where(all_covered)[0]
                
                swap_found = False
                for k in potential_k:
                    if k not in chosen_set:
                        # Perform swap
                        chosen_set.remove(j)
                        chosen_set.add(k)
                        
                        I_chosen[j] = False
                        I_chosen[k] = True
                        
                        K -= A[:, j]
                        K += A[:, k]
                        
                        improved = True
                        swap_found = True
                        break
                        
                if swap_found:
                    break # Restart outer loop
                    
        return list(chosen_set)

    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 15.0:
        code = greedy_vectorized(seed)
        code = local_search_vectorized(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        seed += 1
        
    if best_code is None:
        code = greedy_vectorized(0)
        best_code = local_search_vectorized(code)
        
    return [all_centers[i] for i in best_code]
