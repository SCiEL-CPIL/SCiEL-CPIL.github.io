import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autocorrelation of x."""
    # Use FFT for efficiency
    n = len(x)
    size = 2 * n
    f = np.fft.fft(x, n=size)
    ac = np.fft.ifft(f * np.conj(f)).real
    # The full autocorrelation: ac[k] = sum_i x[i] * x[i-k] for k >= 0
    # numpy's ifft gives circular correlation, we need linear
    # Actually, let's use np.correlate for correctness
    return np.correlate(x, x, mode='full')

def _objective(x):
    """Compute max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(x)
    m = np.max(np.abs(ac))
    return m / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_x = np.ones(N)
    best_obj = _objective(best_x)
    
    # Try a few structured initializations
    candidates = []
    candidates.append(np.ones(N))
    candidates.append(-np.ones(N))
    
    # Random ±1 sequences
    rng = np.random.default_rng(42)
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Smooth sequences: cosine windows
    t = np.linspace(0, 1, N)
    candidates.append(np.cos(2 * np.pi * t))
    candidates.append(np.cos(2 * np.pi * t) ** 2)
    candidates.append(1.0 + 0.5 * np.cos(2 * np.pi * t))
    
    for cand in candidates:
        obj = _objective(cand)
        if obj < best_obj:
            best_obj = obj
            best_x = cand.copy()
    
    # Local search: flip random elements to improve
    x = best_x.copy()
    current_obj = best_obj
    
    # Determine if we should do local search
    if time.time() < deadline - 10:
        # Local search on continuous values
        lr = 0.1
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            idx = rng.integers(0, N)
            old_val = x[idx]
            x[idx] += rng.normal(0, lr)
            
            new_obj = _objective(x)
            if new_obj < current_obj:
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_x = x.copy()
                # Keep the change
            else:
                # Revert
                x[idx] = old_val
        
        # Second phase: finer local search
        if time.time() < deadline - 5:
            lr = 0.01
            for iteration in range(5000):
                if time.time() > deadline - 2:
                    break
                
                idx = rng.integers(0, N)
                old_val = x[idx]
                x[idx] += rng.normal(0, lr)
                
                new_obj = _objective(x)
                if new_obj < current_obj:
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_x = x.copy()
                else:
                    x[idx] = old_val
    
    return best_x
# EVOLVE-BLOCK-END
