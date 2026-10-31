import numpy as np
import time

N = 400
_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def construct():
    """
    Minimizes max(|autoconv(f)|) / (sum(f))^2 using local search on binary sequences.
    """
    # Initialize with all ones
    f = np.ones(N, dtype=np.int8)
    S = np.sum(f)
    
    # Compute autoconvolution
    # C[k] = sum_{i+j=k} f[i] f[j]
    # Length is 2N - 1
    C = np.convolve(f, f)
    
    max_abs_C = np.max(np.abs(C))
    current_score = max_abs_C / (S * S)
    
    best_f = f.copy()
    best_score = current_score
    
    # Local search
    improved = True
    while improved:
        improved = False
        # Check time
        if time.time() - _start_time > _TIME_LIMIT:
            break
            
        # Try flipping each bit
        for i in range(N):
            # Calculate new sum
            new_S = S - 2 * f[i]
            if new_S == 0:
                continue # Avoid division by zero or huge score
            
            # Calculate change in C
            # delta_C[k] = -4 * f[i] * f[k-i] if i != k-i
            # delta_C[k] = -2 if i == k-i
            # We can compute new_C = C + delta_C
            # But we only need max(|new_C|)
            
            # Efficiently compute max(|C + delta_C|)
            # delta_C is non-zero only at indices k where f[k-i] is defined.
            # k ranges from 0 to 2N-2.
            # f[k-i] is defined for 0 <= k-i < N => i <= k < i+N.
            
            # Let's just construct the delta vector for this i
            # This is O(N)
            # But doing it in Python loop for 400 items might be slow.
            # Let's try to vectorize the evaluation of all flips?
            # Hard to vectorize max(|C + delta_i|) for all i at once.
            
            # Let's stick to the loop but optimize the inner part.
            
            # Indices of C affected:
            # k such that 0 <= k-i < N
            # k_min = max(0, i)
            # k_max = min(2N-2, i+N-1)
            
            # Actually, let's just update the C array in place and restore?
            # Or compute new_max directly.
            
            # Let's use a helper to compute new max efficiently.
            # We know C. We know f[i].
            # The terms changing are C[k] - 2*f[i]*f[k-i] (for j=i part)
            # and C[k] - 2*f[k-i]*f[i] (for j=k-i part).
            # So C_new[k] = C[k] - 2*f[i]*(f[k-i] + f[i-k]) ?? No.
            
            # Let's re-derive carefully.
            # C[k] = sum_j f[j] f[k-j]
            # Term with j=i: f[i] f[k-i]
            # Term with j=k-i: f[k-i] f[i]
            # If i != k-i, both exist. Sum = 2 f[i] f[k-i].
            # If i == k-i, only one term f[i]^2 = 1.
            
            # New f[i] = -f[i].
            # New term j=i: -f[i] f[k-i]
            # New term j=k-i: f[k-i] (-f[i]) = -f[k-i] f[i]
            
            # Change for i != k-i:
            # Old: 2 f[i] f[k-i]
            # New: -2 f[i] f[k-i]
            # Delta: -4 f[i] f[k-i]
            
            # Change for i == k-i (k=2i):
            # Old: 1
            # New: 1 (since (-1)^2 = 1)
            # Delta: 0 ??
            # Wait. f[i] is +/- 1. f[i]^2 is always 1.
            # So C[2i] does NOT change when flipping f[i]?
            # Let's check.
            # C[2i] includes term f[i]f[i] = 1.
            # If f[i] becomes -1, term is (-1)(-1) = 1.
            # So C[2i] is invariant to flipping f[i]?
            # Yes, for the diagonal term.
            # But what about other terms in C[2i]?
            # C[2i] = sum_j f[j] f[2i-j].
            # Terms are f[0]f[2i], f[1]f[2i-1], ..., f[i]f[i], ..., f[2i]f[0].
            # Only the middle term f[i]f[i] involves f[i] twice?
            # No, f[j] and f[2i-j].
            # If j=i, term is f[i]f[i].
            # If j != i, term is f[j]f[2i-j]. Neither is f[i] unless j=i or 2i-j=i.
            # So yes, only the term f[i]f[i] changes, and it stays 1.
            # So C[2i] is unchanged.
            
            # So for k != 2i, Delta C[k] = -4 f[i] f[k-i] (if k-i in range).
            # For k = 2i, Delta C[k] = 0.
            
            # So we can compute the new max by checking the affected indices.
            
            # Affected indices k:
            # 0 <= k < 2N
            # k-i in [0, N) => i <= k < i+N
            # Also k-i != i => k != 2i
            
            # Let's create a mask of affected indices.
            # This is still O(N) per flip.
            
            # To speed up, we can precompute the "impact" of each flip?
            # No, C changes.
            
            # Let's just do the O(N) update in NumPy.
            
            # Range of k
            k_start = max(0, i)
            k_end = min(2*N - 1, i + N)
            
            # Indices in C
            idx = np.arange(k_start, k_end)
            
            # Corresponding indices in f: k - i
            f_idx = idx - i
            
            # Delta C for these indices
            # delta = -4 * f[i] * f[f_idx]
            # Except where k == 2i (f_idx == i), delta is 0.
            
            delta = -4 * f[i] * f[f_idx]
            
            # Mask out the center
            center_mask = (idx == 2 * i)
            delta[center_mask] = 0
            
            # New C values
            C_new_vals = C[idx] + delta
            
            # Max of new C values
            local_max = np.max(np.abs(C_new_vals))
            
            # Max of unchanged C values
            # We need global max.
            # max(|C_new|) = max(local_max, max(|C[~mask]|))
            # Computing max(|C[~mask]|) is O(N) if not precomputed.
            # We can maintain a histogram of |C| values?
            # Or just compute global max of C_new?
            # C_new is C with a few elements changed.
            # We can copy C, update, and max.
            # Copy is O(N). Update is O(N). Max is O(N).
            # Total O(N).
            
            # Let's just copy C to C_temp
            C_temp = C.copy()
            C_temp[idx] = C_new_vals
            
            new_max_abs_C = np.max(np.abs(C_temp))
            
            new_score = new_max_abs_C / (new_S * new_S)
            
            if new_score < current_score:
                # Accept
                f[i] = -f[i]
                S = new_S
                C = C_temp
                max_abs_C = new_max_abs_C
                current_score = new_score
                improved = True
                best_f = f.copy()
                best_score = current_score

    return best_f

# EVOLVE-BLOCK-END
