"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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

def _optimize_signs(indices, h_vals, time_limit=5.0):
    """
    Given a fixed set of active indices, optimize the signs (+1/-1)
    to minimize the peak autocorrelation.
    """
    n = len(h_vals)
    if n == 0:
        return h_vals, 0.0
    
    # Initialize h
    h = np.zeros(N)
    for idx, val in zip(indices, h_vals):
        h[idx] = val
    
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    # Greedy sign flipping
    # We iterate until no improvement or time limit
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try flipping each sign
        # To be faster, we can shuffle the order
        order = np.random.permutation(n)
        for i in order:
            idx = indices[i]
            # Flip sign
            trial_h = best_h.copy()
            trial_h[idx] = -trial_h[idx]
            s = _score(trial_h)
            if s < best_score - 1e-12:
                best_h = trial_h
                best_score = s
                improved = True
                # Restart the loop to try other flips from the new state
                # Or just break and let the outer while loop restart
                break
                
    return best_h, best_score

def _local_search_indices(h, time_limit=10.0):
    """
    Local search that allows changing the set of active indices.
    Moves:
    1. Flip sign of an active index.
    2. Zero out an active index.
    3. Activate a zero index with +1 or -1.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    while (time.time() - start) < time_limit:
        improved = False
        
        # Get current active and zero indices
        active_idx = np.nonzero(best_h)[0]
        zero_idx = np.nonzero(best_h == 0)[0]
        
        if len(active_idx) == 0:
            break
            
        # 1. Flip signs
        # Try a random subset to save time, or all if sparse
        if len(active_idx) < 100:
            order = active_idx
        else:
            order = np.random.choice(active_idx, size=50, replace=False)
            
        for i in order:
            trial = best_h.copy()
            trial[i] = -trial[i]
            s = _score(trial)
            if s < best_score - 1e-12:
                best_h = trial
                best_score = s
                improved = True
                break
        
        # 2. Zero out
        if not improved:
            if len(active_idx) > 10: # Don't zero out if too sparse
                if len(active_idx) < 100:
                    order = active_idx
                else:
                    order = np.random.choice(active_idx, size=50, replace=False)
                for i in order:
                    trial = best_h.copy()
                    trial[i] = 0.0
                    s = _score(trial)
                    if s < best_score - 1e-12:
                        best_h = trial
                        best_score = s
                        improved = True
                        break
        
        # 3. Activate zeros
        if not improved:
            if len(zero_idx) > 0:
                # Try a few random zeros
                k = min(20, len(zero_idx))
                order = np.random.choice(zero_idx, size=k, replace=False)
                for i in order:
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
            # If no single move improves, try a random restart or break
            # To avoid getting stuck, we can do a small random perturbation
            # But for now, break to save time for other restarts
            break
            
    return best_h, best_score

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Initialize with sparse random patterns of varying density.
    2. Optimize signs for each pattern.
    3. Perform local search on indices and signs.
    4. Return the best solution found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Try different sparse densities
    # Sparse signals can have very low autocorrelation if indices are well-chosen
    densities = [10, 20, 30, 40, 50, 60, 80, 100, 150]
    
    for k in densities:
        if time.time() > deadline - 15:
            break
            
        # Multiple random restarts for each density
        # Limit restarts based on remaining time
        max_restarts = 5 if k < 50 else 3
        for trial in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Random selection of k indices
            indices = np.random.choice(N, size=k, replace=False)
            # Random initial signs
            signs = np.random.choice([-1.0, 1.0], size=k)
            
            h = np.zeros(N)
            h[indices] = signs
            
            # Phase 1: Optimize signs for this fixed support
            h_opt, score = _optimize_signs(indices, signs, time_limit=2.0)
            
            # Phase 2: Local search on indices and signs
            if time.time() < deadline - 10:
                h_opt, score = _local_search_indices(h_opt, time_limit=5.0)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
    # If we found a good solution, do a final polish
    if best_h is not None and time.time() < deadline - 10:
        # Final polish with a longer local search
        best_h, best_score = _local_search_indices(best_h, time_limit=15.0)
        
    # Fallback: if no sparse solution was found or if a full sequence is better
    # Try a full +/- 1 sequence with a known low-autocorrelation pattern
    if best_h is None or best_score > 20:
        # Use a random full sequence
        h_full = np.random.choice([-1, 1], size=N).astype(np.float64)
        h_full_opt, score_full = _local_search_indices(h_full, time_limit=10.0)
        
        if score_full < best_score:
            best_score = score_full
            best_h = h_full_opt.copy()
    
    # Ensure we have a valid return
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
