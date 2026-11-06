"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_value(h):
    """Compute the objective value."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    mx = np.max(conv)
    return abs(2.0 * N * mx / (s * s))

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin
    
    # Strategy: Use a pattern with strong negative auto-correlation
    # Start with alternating pattern and optimize
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # = 2.0
    
    # Try several initial patterns
    patterns = []
    
    # Pattern 1: Alternating with optimized ratio
    for c in [0.5, 0.8, 0.9, 0.95, 0.99]:
        h = np.ones(N)
        h[1::2] = -c
        patterns.append(h.copy())
    
    # Pattern 2: Block alternating
    for block in [2, 4, 8, 16]:
        h = np.ones(N)
        for i in range(N):
            if (i // block) % 2 == 1:
                h[i] = -0.5
        patterns.append(h.copy())
    
    # Pattern 3: Smooth oscillation
    for freq in [0.5, 1.0, 2.0, 5.0, 10.0]:
        x = np.linspace(0, freq * 2 * np.pi, N)
        patterns.append(1.0 + 0.5 * np.sin(x))
    
    # Pattern 4: Random with bias
    rng = np.random.RandomState(42)
    for _ in range(5):
        h = rng.randn(N)
        h = h - np.mean(h)  # zero mean component
        h = 1.0 + 0.3 * h / (np.std(h) + 1e-10)
        patterns.append(h)
    
    # Evaluate all patterns
    for h in patterns:
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Now optimize from the best starting point
    if time.time() < deadline - 30:
        def objective(x):
            return _compute_value(x)
        
        result = minimize(objective, best_h, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success or result.fun < best_val:
            if np.all(np.isfinite(result.x)) and abs(np.sum(result.x)) > 1e-10:
                best_h = result.x
                best_val = result.fun
        
        # Multiple restarts
        n_restarts = 0
        while time.time() < deadline - 10 and n_restarts < 20:
            n_restarts += 1
            rng = np.random.RandomState(n_restarts * 12345 + 67890)
            # Perturb the best solution
            h_init = best_h + 0.1 * rng.randn(N)
            
            res = minimize(objective, h_init, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                if np.all(np.isfinite(res.x)) and abs(np.sum(res.x)) > 1e-10:
                    best_h = res.x
                    best_val = res.fun
    
    # Final check
    final_val = _compute_value(best_h)
    if final_val < 2.0:  # Better than seed
        return best_h
    else:
        return np.ones(N)
# EVOLVE-BLOCK-END
