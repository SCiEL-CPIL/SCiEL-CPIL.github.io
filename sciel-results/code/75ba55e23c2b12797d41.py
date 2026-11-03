"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autoconv(x):
    """Compute full autoconvolution array for ±1 signal x."""
    n = len(x)
    x_padded = np.zeros(2 * n)
    x_padded[:n] = x
    X = np.fft.fft(x_padded, n=2 * n)
    C = np.fft.ifft(X * X.conj()).real
    # C[n+k] = sum_i x[i] * x[i+k] for k in [-(n-1), n-1]
    return C

def _max_abs_ratio(C, S2):
    if S2 == 0:
        return np.inf
    return np.max(np.abs(C)) / S2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with all +1
    best_x = np.ones(N, dtype=np.int8)
    best_C = _compute_autoconv(best_x)
    best_S = np.sum(best_x)
    best_S2 = best_S * best_S
    best_score = _max_abs_ratio(best_C, best_S2)
    
    np.random.seed(42)
    
    for init_iter in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization with bias toward +1
        x = np.random.choice([1, -1], size=N, p=[0.8, 0.2])
        x = x.astype(np.int8)
        
        C = _compute_autoconv(x)
        S = np.sum(x)
        S2 = S * S
        current_score = _max_abs_ratio(C, S2)
        
        if current_score < best_score:
            best_x = x.copy()
            best_C = C.copy()
            best_S = S
            best_S2 = S2
            best_score = current_score
        
        # Simulated annealing with incremental updates
        T = 0.005
        T_min = 1e-7
        cooling = 0.98
        
        for step in range(3000):
            if time.time() > deadline:
                break
            if T < T_min:
                break
            
            idx = np.random.randint(N)
            old_val = x[idx]
            x[idx] = -old_val
            
            # Update autoconvolution incrementally
            # C(k) = sum_i x[i] * x[i+k]
            # When x[idx] changes from old_val to -old_val (change of -2*old_val):
            # C(k) changes by: -2*old_val * x[idx+k] for each valid k
            
            # For lag k: C(k) = sum_i x[i]*x[i+k]
            # The term involving x[idx] appears when i=idx (x[idx]*x[idx+k]) or i+k=idx (x[i]*x[idx])
            # So C(k) gets contribution from x[idx]*x[idx+k] (when i=idx) and x[idx-k]*x[idx] (when i+k=idx, i=idx-k)
            
            # Let's update C directly
            # C is stored as C[n+k] for lag k
            # For each k where idx+k is in [0, N-1]: C[k] -= 2*old_val*x[idx+k]
            # For each k where idx-k is in [0, N-1]: C[k] -= 2*old_val*x[idx-k]
            
            # Actually, let me just recompute C since N=400 is small
            # FFT of size 800 is fast enough
            C = _compute_autoconv(x)
            S = S - 2 * old_val  # Since x[idx] changed by -2*old_val
            S2 = S * S
            
            new_score = _max_abs_ratio(C, S2)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-12)):
                current_score = new_score
                if current_score < best_score:
                    best_x = x.copy()
                    best_C = C.copy()
                    best_S = S
                    best_S2 = S2
                    best_score = current_score
            else:
                x[idx] = old_val  # Revert
            
            T *= cooling
    
    return best_x
# EVOLVE-BLOCK-END
