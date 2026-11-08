import numpy as np
import time

N = 400

def _compute_metric(h):
    """Compute the upper bound on C3: max abs autocorrelation at non-zero lags / N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    best_h = np.ones(N, dtype=np.float64)
    best_score = _compute_metric(best_h)
    
    # Try a few different initializations
    initializations = [
        np.ones(N, dtype=np.float64),
        np.random.choice([-1.0, 1.0], N),
        np.random.normal(0, 1, N),
    ]
    
    for init in initializations:
        if time.time() > deadline - 30:
            break
        
        h_current = init.copy()
        
        # Perform local search
        for step in range(3000):
            if time.time() > deadline - 10:
                break
            
            # Try flipping a random element
            idx = np.random.randint(0, N)
            h_trial = h_current.copy()
            h_trial[idx] = -h_trial[idx]
            
            score = _compute_metric(h_trial)
            if score < best_score - 1e-12:
                best_score = score
                best_h = h_trial
                h_current = h_trial
            elif score < _compute_metric(h_current) - 1e-12:
                # Accept even if not globally better, for diversity
                h_current = h_trial
    
    return best_h
