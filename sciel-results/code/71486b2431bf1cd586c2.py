"""Improved program for the third autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _value(h):
    """Compute the evaluator's value: abs(2*N*max(convolve(h,h))/sum(h)^2)."""
    s = np.sum(h)
    if s == 0:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with the seed solution
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = _value(best_h)  # should be 2.0
    
    # Try a few structured patterns
    # Pattern 1: alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1.0
    v = _value(h_alt)
    if v < best_val:
        best_val = v
        best_h = h_alt.copy()
    
    # Pattern 2: block pattern (first half positive, second half negative)
    h_block = np.ones(N)
    h_block[N//2:] = -1.0
    v = _value(h_block)
    if v < best_val:
        best_val = v
        best_h = h_block.copy()
    
    # Pattern 3: sinusoidal
    x = np.linspace(0, 2*np.pi, N)
    h_sin = np.cos(x)
    v = _value(h_sin)
    if v < best_val:
        best_val = v
        best_h = h_sin.copy()
    
    # Pattern 4: two blocks
    h_2block = np.ones(N)
    h_2block[N//4:3*N//4] = -1.0
    v = _value(h_2block)
    if v < best_val:
        best_val = v
        best_h = h_2block.copy()
    
    # Pattern 5: random initialization with optimization
    np.random.seed(42)
    h_rand = np.random.randn(N)
    v = _value(h_rand)
    if v < best_val:
        best_val = v
        best_h = h_rand.copy()
    
    # Try gradient-based optimization starting from best found so far
    def objective(x):
        return _value(x)
    
    def gradient(x):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            g[i] = (objective(xp) - objective(xm)) / (2*eps)
        return g
    
    # Use L-BFGS-B which can handle this
    try:
        res = minimize(objective, best_h, method='L-BFGS-B', 
                       options={'maxiter': 200, 'gtol': 1e-5, 'disp': False})
        if time.time() < deadline:
            v = _value(res.x)
            if v < best_val:
                best_val = v
                best_h = res.x.copy()
    except Exception:
        pass
    
    # Try a few more random restarts if time permits
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 20:
            break
        np.random.seed(seed)
        h_init = np.random.randn(N)
        # Normalize to avoid numerical issues
        h_init = h_init / np.sqrt(np.sum(h_init**2)) * 10
        
        try:
            res = minimize(objective, h_init, method='L-BFGS-B',
                          options={'maxiter': 100, 'gtol': 1e-5, 'disp': False})
            if time.time() > deadline - 10:
                break
            v = _value(res.x)
            if v < best_val:
                best_val = v
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Final safety check
    if time.time() > deadline:
        return np.ones(N)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
