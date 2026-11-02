"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _compute_score(U):
    """
    Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    U is a list of nonnegative integers containing 0.
    Uses FFT-based convolution for efficiency.
    """
    if len(U) < 2:
        return 0.0
    max_U = max(U)
    if max_U == 0:
        return 0.0
    
    # Create indicator vector for U
    M = max_U
    # For sumset: we need values from 0 to 2*M
    # For difference set: we need values from -M to M, shift by M to get 0 to 2*M
    
    # Sumset via convolution
    # u[i] = 1 if i in U else 0
    u = np.zeros(2 * M + 1, dtype=np.float64)
    for x in U:
        u[x] = 1.0
    
    # Convolve u with itself to get sumset
    # The sumset U+U has elements x+y where x,y in U
    # Using FFT
    n_fft = 1
    while n_fft < 2 * (2 * M + 1):
        n_fft *= 2
    
    U_fft = np.fft.fft(u, n=n_fft)
    conv = np.fft.ifft(U_fft * U_fft).real
    
    # conv[i] > 0.5 means i is in U+U
    sumset_count = np.sum(conv > 0.5)
    
    # Difference set U-U
    # u_diff[x] = 1 if x in U else 0
    # (U-U) contains x-y for x,y in U
    # This is equivalent to: for each d, exists x,y in U such that x-y=d
    # We can compute this by: conv_diff[i] = sum over x of u[x] * u[x - i] for i >= 0
    # Actually, let's use a different approach.
    # U-U = {x-y : x,y in U}
    # Let v[x] = 1 if x in U else 0
    # Then (U-U)[d] = sum_x v[x] * v[x-d] for d >= 0 (this gives x-y=d)
    # For d < 0, (U-U)[d] = sum_x v[x] * v[x-d] = sum_y v[y+d] * v[y]
    # 
    # Alternative: U-U is the support of the "cross-correlation" of v with itself.
    # Let's just compute it directly using the fact that U-U is symmetric.
    # |U-U| = |U+(-U)|
    # Let w[x] = 1 if -x in U else 0, i.e., w[x] = v[-x]
    # Then U+(-U) can be computed by convolving v with w.
    
    # w[x] = 1 if -x in U. Since U contains nonnegative integers, -x in U means x=0 (since -x <= 0 and U >= 0).
    # Wait, that's not right. Let me reconsider.
    
    # Actually, let's just compute |U-U| directly.
    # U-U = {a-b : a in U, b in U}
    # The range is from -max_U to max_U.
    # We can create an array of size 2*max_U+1 indexed from 0 to 2*max_U,
    # where index i corresponds to value i - max_U.
    
    # For each pair (a, b), a-b is in range [-max_U, max_U].
    # Using convolution: let v[a] = 1 if a in U.
    # Let w[b] = 1 if b in U.
    # Then (v * w)[k] = sum_j v[j] * w[k-j]
    # We want to know for each d, if there exists a,b with a-b=d, i.e., a = b+d.
    # So sum_b v[b+d] * w[b] = sum_b v[b+d] * v[b]
    # This is the cross-correlation of v with itself.
    
    # Cross-correlation of v with itself:
    # corr[d] = sum_b v[b] * v[b+d] for d >= 0
    # For d < 0, corr[d] = corr[-d] by symmetry of the set of differences (if d is a difference, so is -d).
    # Actually, U-U is always symmetric: if d = a-b is in U-U, then -d = b-a is in U-U.
    # So |U-U| = 1 + 2 * |{d > 0 : d in U-U}|
    
    # Let's compute the cross-correlation using FFT.
    # corr[d] for d >= 0: sum_b v[b] * v[b+d]
    # This is the same as the convolution of v with a reversed version of v.
    
    # Let's define v_rev[x] = v[max_U - x]
    # Then (v * v_rev)[k] = sum_j v[j] * v_rev[k-j] = sum_j v[j] * v[max_U - (k-j)] = sum_j v[j] * v[max_U - k + j]
    # Let d = max_U - k, so k = max_U - d
    # (v * v_rev)[max_U - d] = sum_j v[j] * v[d + j]
    # This is corr[d] for d >= 0.
    
    v = np.zeros(2 * M + 1, dtype=np.float64)
    for x in U:
        v[x] = 1.0
    
    v_rev = v[::-1].copy()
    
    n_fft2 = 1
    while n_fft2 < 2 * (2 * M + 1):
        n_fft2 *= 2
    
    V_fft = np.fft.fft(v, n=n_fft2)
    Vrev_fft = np.fft.fft(v_rev, n=n_fft2)
    corr = np.fft.ifft(V_fft * Vrev_fft).real
    
    # corr[max_U - d] = sum_j v[j] * v[j + d] for d >= 0
    # So for d in [0, M], the count of pairs (j, j+d) both in U is corr[M - d]
    # The difference d is in U-U if corr[M - d] > 0.5
    
    # d ranges from 0 to M
    # d=0: always in U-U (0 = a-a)
    # d > 0: in U-U if corr[M - d] > 0.5
    
    diff_pos = np.sum(corr[:M+1] > 0.5)  # d from 0 to M, corr index M-d, so corr[0] to corr[M]
    # Wait, corr[M - d] for d=0 is corr[M], for d=M is corr[0].
    # So corr[0] to corr[M] corresponds to d=M down to d=0.
    # diff_pos counts the number of d in [0, M] such that d is in U-U.
    
    # |U-U| = 1 + 2 * (diff_pos - 1) = 2 * diff_pos - 1
    diff_count = 2 * diff_pos - 1
    
    ratio = diff_count / sumset_count
    if sumset_count <= 0:
        return 0.0
    
    import math
    log_denom = math.log(2 * max_U + 1)
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / log_denom


def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Start with a reasonable initial set
    # The seed is [0, 1, 3]
    current_set = [0, 1, 3]
    current_score = _compute_score(current_set)
    if current_score > best_score:
        best_score = current_score
        best_set = list(current_set)
    
    # Local search with random restarts
    random.seed(42)
    
    # Try different sizes and max values
    for n in [5, 7, 10, 15, 20, 30, 40, 50]:
        if time.time() - start_time > deadline - 10:
            break
        
        # Try several random restarts for this size
        for restart in range(5):
            if time.time() - start_time > deadline - 5:
                break
            
            # Generate a random set of size n containing 0
            # Choose max_U based on n
            # Heuristic: max_U around n * (n+1) / 2 or similar
            max_val = n * (n + 1) // 2 + random.randint(0, n)
            
            # Random set
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            U = [0] + sorted(candidates[:n-1])
            
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_set = list(U)
            
            # Local search: try adding, removing, or swapping elements
            improved = True
            local_iter = 0
            while improved and local_iter < 50 and time.time() - start_time < deadline - 2:
                improved = False
                local_iter += 1
                
                # Try adding a random element
                if len(U) < n + 5:
                    new_val = random.randint(1, max_val + 10)
                    if new_val not in U:
                        U_new = sorted(U + [new_val])
                        s_new = _compute_score(U_new)
                        if s_new > score:
                            U = U_new
                            score = s_new
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_set = list(U)
                
                # Try removing an element (not 0)
                if len(U) > 2:
                    idx = random.randint(1, len(U) - 1)
                    U_new = U[:idx] + U[idx+1:]
                    s_new = _compute_score(U_new)
                    if s_new > score:
                        U = U_new
                        score = s_new
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_set = list(U)
                
                # Try swapping an element
                if len(U) > 2:
                    idx = random.randint(1, len(U) - 1)
                    old_val = U[idx]
                    new_val = random.randint(1, max_val + 10)
                    if new_val != old_val and new_val not in U:
                        U_new = sorted(U[:idx] + [new_val] + U[idx+1:])
                        s_new = _compute_score(U_new)
                        if s_new > score:
                            U = U_new
                            score = s_new
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_set = list(U)
    
    # Final polish: try to improve the best set found
    if time.time() - start_time < deadline - 5:
        U = list(best_set)
        score = best_score
        improved = True
        while improved and time.time() - start_time < deadline - 1:
            improved = False
            # Try small perturbations
            for _ in range(20):
                if time.time() - start_time > deadline - 1:
                    break
                
                # Add a small random element
                max_U = max(U)
                new_val = random.randint(1, max_U + 5)
                if new_val not in U:
                    U_new = sorted(U + [new_val])
                    s_new = _compute_score(U_new)
                    if s_new > score:
                        U = U_new
                        score = s_new
                        improved = True
                
                # Remove a random non-zero element
                if len(U) > 2:
                    idx = random.randint(1, len(U) - 1)
                    U_new = U[:idx] + U[idx+1:]
                    s_new = _compute_score(U_new)
                    if s_new > score:
                        U = U_new
                        score = s_new
                        improved = True
        
        if score > best_score:
            best_score = score
            best_set = U
    
    return best_set

# EVOLVE-BLOCK-END
