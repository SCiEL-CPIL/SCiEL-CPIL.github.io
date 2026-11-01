"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr(x):
    """Compute full linear autocorrelation of a 1D array."""
    n = len(x)
    corr = np.zeros(2 * n - 1)
    for k in range(n):
        corr[k] = np.dot(x[:n-k], x[k:])
        corr[2*n-1-k] = corr[k]
    return corr

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Initialize with random ±1 sequence
    rng = np.random.RandomState(42)
    x = rng.choice([-1, 1], size=N)
    
    # Compute initial autocorrelation
    corr = _compute_autocorr(x)
    current_max = np.max(np.abs(corr))
    
    best_x = x.copy()
    best_max = current_max
    
    # Simulated annealing parameters
    T = 5.0
    T_min = 0.005
    cooling_rate = 0.9995
    
    k_vals = np.arange(N)
    
    for it in range(2000000):
        if time.time() > deadline:
            break
        
        # Choose a random position to flip
        j = rng.randint(0, N)
        old_val = x[j]
        
        # Compute delta_R for lags 0..N-1
        # For lag k: delta_R[k] = -2 * old_val * (x[j+k] if j+k<N else 0 + x[j-k] if j-k>=0 else 0)
        
        # Vectorized computation
        delta_R_half = np.zeros(N)
        
        # term1: x[j+k] for k where j+k < N, i.e., k < N-j
        if j < N:
            idx1 = np.arange(0, N - j)
            delta_R_half[idx1] += -2 * old_val * x[j + idx1]
        
        # term2: x[j-k] for k where j-k >= 0, i.e., k <= j
        if j >= 0:
            idx2 = np.arange(0, j + 1)
            delta_R_half[idx2] += -2 * old_val * x[j - idx2]
        
        # Build full delta_R
        delta_R = np.zeros(2*N-1)
        delta_R[:N] = delta_R_half
        delta_R[N:] = delta_R_half[N-1:0:-1]
        
        new_corr = corr + delta_R
        new_max = np.max(np.abs(new_corr))
        
        # Acceptance criterion
        delta = new_max - current_max
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            x[j] = -x[j]
            corr = new_corr
            current_max = new_max
            
            if current_max < best_max:
                best_max = current_max
                best_x = x.copy()
        
        # Cool down
        T = max(T * cooling_rate, T_min)
    
    return best_x
# EVOLVE-BLOCK-END
