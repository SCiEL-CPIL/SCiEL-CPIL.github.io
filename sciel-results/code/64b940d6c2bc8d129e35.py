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
    # Simulated Annealing approach
    # This allows escaping local minima where lag-0 is the maximum autocorrelation peak.
    
    rng = np.random.default_rng(12345)
    
    # Start with a random configuration, possibly with some structure
    # We want to encourage non-zero lags to be large.
    # Initialize with a low-frequency cosine + noise
    t = np.linspace(0, 2*np.pi, N)
    h = 1.0 + 0.8 * np.cos(t) + 0.1 * rng.normal(0, 1, N)
    
    current_val = _objective(h)
    best_h = h.copy()
    best_val = current_val
    
    # SA Parameters
    T_start = 1.0
    T_end = 1e-4
    num_iterations = 20000
    cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
    
    # We will perturb a small number of indices at a time
    # This is faster than full vector perturbation
    
    for i in range(num_iterations):
        # Perturb 1-5 random indices
        num_perturb = rng.integers(1, 6)
        indices = rng.integers(0, N, num_perturb)
        
        h_new = h.copy()
        for idx in indices:
            h_new[idx] += rng.normal(0, 0.5)
            
        val_new = _objective(h_new)
        
        # Avoid division by zero issues in the objective if sum is close to 0
        # _objective returns 1e12 if sum is 0, which is effectively infinite penalty
        
        delta = val_new - current_val
        
        # Acceptance criterion
        if delta < 0:
            # Better solution, always accept
            h = h_new
            current_val = val_new
        else:
            # Worse solution, accept with probability exp(-delta / T)
            # If delta is large, probability is small
            # Clamp T to avoid overflow in exp
            T = T_start * (cooling_rate ** i)
            if T > 1e-10:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    h = h_new
                    current_val = val_new
        
        # Track best
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # Periodic restart if stuck? 
        # For now, just let it run.
        
    return best_h
# EVOLVE-BLOCK-END
