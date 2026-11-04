"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak(h):
    """Compute max absolute autocorrelation at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[k] for k in 1..n-1 gives positive lags
    # For real signals, acf is symmetric, so max over 1..n-1 suffices
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _greedy_local_search(h, max_iters=2000, time_limit=10.0):
    """Greedy local search: flip signs or set to zero to reduce peak."""
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        improved = False
        # Try flipping signs of non-zero entries
        nonzero_idx = np.nonzero(best_h)[0]
        if len(nonzero_idx) == 0:
            break
        
        # Shuffle order for diversity
        order = np.random.permutation(len(nonzero_idx))
        for idx_pos in order:
            i = nonzero_idx[idx_pos]
            # Try flipping sign
            trial = best_h.copy()
            trial[i] = -trial[i]
            s = _score(trial)
            if s < best_score - 1e-12:
                best_h = trial
                best_score = s
                improved = True
                break
        
        # Try zeroing out entries
        nonzero_idx = np.nonzero(best_h)[0]
        if len(nonzero_idx) == 0:
            break
        order = np.random.permutation(len(nonzero_idx))
        for idx_pos in order:
            i = nonzero_idx[idx_pos]
            trial = best_h.copy()
            trial[i] = 0.0
            s = _score(trial)
            if s < best_score - 1e-12:
                best_h = trial
                best_score = s
                improved = True
                break
        
        # Try activating zero entries
        zero_idx = np.nonzero(best_h == 0)[0]
        if len(zero_idx) > 0:
            order = np.random.permutation(len(zero_idx))
            for idx_pos in order[:50]:  # Limit activations per iteration
                i = zero_idx[idx_pos]
                for sign in [1.0, -1.0]:
                    trial = best_h.copy()
                    trial[i] = sign
                    s = _score(trial)
                    if s < best_score - 1e-12:
                        best_h = trial
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_h, best_score

def _try_sparse_pattern(k, signs_init, time_limit=8.0):
    """Try a sparse pattern with k non-zero entries at specific positions."""
    h = np.zeros(N)
    h[signs_init[0]:signs_init[0]+k:1] = 0  # placeholder
    # Actually, let's use the given indices
    indices = signs_init
    for i in indices:
        h[i] = 1.0
    
    # Optimize signs and values
    best_h, best_score = _greedy_local_search(h, max_iters=500, time_limit=time_limit)
    return best_h, best_score

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound (max absolute autocorrelation at non-zero lags).
    
    Strategy: Sparse combinatorial design.
    1. Select a small subset of intervals to be non-zero.
    2. Use random permutations to choose active indices.
    3. Optimize signs via greedy local search.
    4. Return the best solution found within time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try different sparse densities
    for k in [8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]:
        if time.time() > deadline - 10:
            break
        
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random selection of k indices
            indices = np.random.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[indices] = 1.0
            
            # Optimize
            h_opt, score = _greedy_local_search(h, max_iters=300, time_limit=5.0)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Final polish on best solution
    if time.time() < deadline - 5:
        best_h, best_score = _greedy_local_search(best_h, max_iters=1000, time_limit=15.0)
    
    return best_h
# EVOLVE-BLOCK-END
