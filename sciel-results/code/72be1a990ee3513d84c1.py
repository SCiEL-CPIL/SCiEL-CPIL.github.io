"""Program exploring non-constant step heights to reduce the third autocorrelation constant."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _objective(h):
    s = np.sum(h)
    if s == 0:
        return 1e12
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))

def construct():
    best_h = np.ones(N)
    best_val = _objective(best_h)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Antisymmetric perturbation
    # h[i] = 1 + A * x[i] where x is antisymmetric: x[i] = -x[N-1-i]
    
    # Generate random antisymmetric base
    # Only need to define first N/2, second half is determined
    half = N // 2
    x_base = rng.standard_normal(half)
    x_full = np.zeros(N)
    x_full[:half] = x_base
    x_full[half:] = -x_base[::-1]
    if N % 2 != 0:
        x_full[half] = 0
        
    # Normalize x to unit variance
    x_full = x_full / np.std(x_full)
    
    # Try different amplitudes
    for A in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        h = 1.0 + A * x_full
        val = _objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Refine the best one with local search
    h = best_h.copy()
    current_val = best_val
    
    # Hill climbing on the antisymmetric component
    # We can update the first half, and mirror the second half
    step_size = 0.1
    for i in range(1000):
        idx = rng.integers(half)
        # Perturb x[idx]
        delta = rng.uniform(-step_size, step_size)
        
        h_new = h.copy()
        # Update symmetric pair
        h_new[idx] += delta
        h_new[N-1-idx] -= delta
        
        val_new = _objective(h_new)
        if val_new < current_val:
            h = h_new
            current_val = val_new
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        else:
            step_size *= 0.95
            if step_size < 1e-4:
                break
                
    return best_h
# EVOLVE-BLOCK-END
