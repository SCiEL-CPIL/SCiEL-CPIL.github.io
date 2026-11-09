import numpy as np
import time

N = 400

def _compute_metric(h):
    """Compute the upper bound on C3: max abs autocorrelation at non-zero lags / N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def _compute_sos_metric(h):
    """Compute sum of squared autocorrelation at non-zero lags."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    return np.sum(R[1:]**2)

def _greedy_polish(h, deadline):
    """Greedy polish: try each position, keep if it improves."""
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(N):
            if time.time() > deadline - 5:
                break
            h_trial = h.copy()
            h_trial[idx] = -h_trial[idx]
            if _compute_metric(h_trial) < _compute_metric(h) - 1e-12:
                h = h_trial
                improved = True
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = None
    best_score = float('inf')
    
    # Multiple restarts with different seeds
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        seed = 42 + restart
        rng = np.random.RandomState(seed)
        
        # Initialize with random ±1 values
        h_current = rng.choice([-1.0, 1.0], N)
        
        # Local search with hybrid moves
        for step in range(5000):
            if time.time() > deadline - 10:
                break
            
            # Choose move type: flip or swap
            move_type = rng.random()
            if move_type < 0.7:
                # Flip
                idx = rng.randint(0, N)
                h_trial = h_current.copy()
                h_trial[idx] = -h_trial[idx]
            else:
                # Swap two positions
                idx1, idx2 = rng.choice(N, size=2, replace=False)
                h_trial = h_current.copy()
                h_trial[idx1], h_trial[idx2] = h_current[idx2], h_current[idx1]
            
            score = _compute_metric(h_trial)
            if score < best_score - 1e-12:
                best_score = score
                best_h = h_trial.copy()
                h_current = h_trial
            elif score < _compute_metric(h_current) - 1e-12:
                h_current = h_trial
        
        # Greedy polish on the best from this restart
        if best_h is not None:
            polished = _greedy_polish(best_h.copy(), deadline)
            polished_score = _compute_metric(polished)
            if polished_score < best_score - 1e-12:
                best_score = polished_score
                best_h = polished
    
    return best_h
