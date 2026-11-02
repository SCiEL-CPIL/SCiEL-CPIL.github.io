import numpy as np
import time

N = 400

def _max_abs_autocorr(s):
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_float = s.astype(np.float64)
    F = np.fft.rfft(s_float, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    if N <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))

def _greedy_improve(s, max_iters=100):
    """Perform greedy local search."""
    s = s.copy()
    current_val = _max_abs_autocorr(s)
    
    for _ in range(max_iters):
        improved = False
        # Random order to break symmetries
        indices = np.random.permutation(N)
        for i in indices:
            old = s[i]
            s[i] = -old
            new_val = _max_abs_autocorr(s)
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                s[i] = old
        if not improved:
            break
    return s, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_seq = None
    best_val = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Initial random starts
    for _ in range(5):
        s = rng.choice([-1, 1], size=N)
        s, val = _greedy_improve(s)
        if val < best_val:
            best_val = val
            best_seq = s.copy()
            
        if time.time() > deadline:
            break

    # Simulated Annealing from the best found so far
    if time.time() < deadline:
        current_seq = best_seq.copy()
        current_val = best_val
        T = 5.0
        cooling = 0.999
        
        while time.time() < deadline:
            idx = rng.integers(0, N)
            old = current_seq[idx]
            current_seq[idx] = -old
            
            new_val = _max_abs_autocorr(current_seq)
            delta = new_val - current_val
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_seq = current_seq.copy()
            else:
                current_seq[idx] = old
                
            T *= cooling
            if T < 0.01:
                T = 0.01

    return best_seq
