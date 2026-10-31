import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Legendre Sequence
    p = 401
    legendre = np.zeros(p, dtype=np.int8)
    for x in range(1, p):
        val = pow(x, (p-1)//2, p)
        legendre[x] = 1 if val == 1 else -1
    seq = legendre[1:p].copy()
    
    # 2. Initial ACF
    # A[k] for k=0..N-1
    # We only care about k=1..N-1
    acf = np.correlate(seq, seq, mode='full')
    # np.correlate(seq, seq, 'full') returns length 2N-1
    # Center is at N-1
    # acf[N-1] is A(0)
    # acf[N-1 + k] is A(k)
    # acf[N-1 - k] is A(-k) = A(k) for real
    # We want max abs of k=1..N-1
    # Indices: N..2N-2 and 0..N-2
    off_peak_acf = np.concatenate([acf[:N-1], acf[N:2*N-1]])
    # Note: acf is symmetric, so we can just take one side if we want, 
    # but max abs is the same.
    
    current_max = np.max(np.abs(off_peak_acf))
    best_seq = seq.copy()
    best_max = current_max
    
    # Helper to update ACF for a flip at index j
    # We need to update the full acf array or just the off-peak parts?
    # Let's maintain the full acf array of length 2N-1.
    # Flip x[j] -> -x[j]
    # Delta at lag k: -2 * x[j] * (x[j-k] + x[j+k])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_idx = -1
        
        for j in range(N):
            # Calculate new max if we flip j
            # We need to simulate the flip on the acf
            # To avoid modifying the original acf, we compute the changes
            
            x_j = seq[j]
            
            # Indices in acf corresponding to lags k
            # Lag k corresponds to acf[N-1+k]
            # The terms affected are lags k where j-k or j+k is in range [0, N-1]
            # i.e., k in [j-(N-1), j] U [j, j+(N-1)] intersect [1, N-1]
            
            # Efficiently compute the new acf values for affected lags
            # We can create a temporary copy of acf? 
            # Copying 800 floats is fast.
            temp_acf = acf.copy()
            
            # Update temp_acf
            # For each lag k, delta = -2 * x_j * (x_{j-k} + x_{j+k})
            # We can vectorize this update.
            
            # Lags k from 1 to N-1
            k_vals = np.arange(1, N)
            
            # Index in seq for x_{j-k}
            idx_m = j - k_vals
            valid_m = (idx_m >= 0) & (idx_m < N)
            # Index in seq for x_{j+k}
            idx_p = j + k_vals
            valid_p = (idx_p >= 0) & (idx_p < N)
            
            x_m = np.zeros(N-1)
            x_p = np.zeros(N-1)
            
            if np.any(valid_m):
                x_m[valid_m] = seq[idx_m[valid_m]]
            if np.any(valid_p):
                x_p[valid_p] = seq[idx_p[valid_p]]
                
            deltas = -2 * x_j * (x_m + x_p)
            
            # Apply deltas to temp_acf
            # acf index for lag k is N-1+k
            temp_acf[N:2*N-1] += deltas
            # acf index for lag -k is N-1-k
            temp_acf[0:N-1] += deltas # Since A(-k)=A(k) and delta is same?
            # Wait, A(-k) = sum x_i x_{i-k}. 
            # Flip x_j affects x_j x_{j-k} in A(k) and x_j x_{j+k} in A(k).
            # In A(-k), the terms are x_{j+k} x_j and x_{j-k} x_j.
            # The delta is the same.
            
            new_max = np.max(np.abs(np.concatenate([temp_acf[:N-1], temp_acf[N:2*N-1]])))
            
            if new_max < best_max - 1e-9:
                best_max = new_max
                best_idx = j
                best_delta = new_max
                # Store the deltas or recompute later? 
                # We can store the new acf
                best_acf = temp_acf
        
        if best_idx != -1:
            seq[best_idx] *= -1
            acf = best_acf
            best_seq = seq.copy()
            improved = True

    return best_seq
