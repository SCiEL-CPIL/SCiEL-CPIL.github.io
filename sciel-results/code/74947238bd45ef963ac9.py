import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_lin(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _metric(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    conv = _autoconv_lin(x)
    return float(np.max(np.abs(conv))) / (s * s)

def _soft_max_abs(conv, beta=500.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-6:
        return 1e10
    conv = _autoconv_lin(x)
    return _soft_max_abs(conv, beta=500.0) / (s * s)

def _construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(2024)
    
    # Initialize with a symmetric, smooth function to help convergence
    # e.g., a raised cosine or similar
    t = np.linspace(0, 1, N)
    x0 = 1.0 + 0.5 * np.cos(2 * np.pi * t)
    
    # Normalize initial sum to be safe, though metric is scale invariant
    # We don't strictly need to fix sum=1 because the objective handles it,
    # but fixing scale can sometimes help optimization landscapes.
    # However, since it's scale invariant, the optimizer might drift.
    # Let's add a soft constraint or just let it be.
    
    # L-BFGS-B doesn't support constraints well without SLSQP.
    # SLSQP is slower.
    # We can re-parameterize to fix the sum?
    # x = y + (1 - sum(y)/N) * ones? No, that's linear.
    
    # Let's just run L-BFGS-B on the raw x.
    # The scale invariance creates a flat direction.
    # To mitigate, we can add a tiny L2 penalty to keep x bounded.
    
    def obj_with_reg(x):
        base = _objective(x)
        reg = 1e-6 * np.sum(x**2)
        return base + reg
        
    # Initial guess
    x_init = x0.copy()
    
    # Try a few different initializations to avoid local minima
    best_x = x_init
    best_val = _metric(x_init)
    
    # Run optimization
    # We need to be careful with the time.
    # L-BFGS-B with finite differences is slow for N=400.
    # We need analytical gradients or a better method.
    
    # Gradient of _objective(x):
    # f(x) = M(x) / S(x)^2
    # M(x) = log-sum-exp(beta * |conv|) (approx)
    # This is complex.
    
    # Alternative: Use Nelder-Mead? No, too slow for 400 dims.
    # Use COBYLA? No.
    
    # Let's try to use the spectral insight to generate a good initial point,
    # then use a simple local search or coordinate descent which is fast.
    
    # The current program uses spectral optimization.
    # The "different" strategy is time-domain epigraph.
    # But if we can't do fast gradients, maybe we can do a fast heuristic search
    # that is structurally different?
    
    # Actually, the instruction says "different design or modelling family".
    # The current one is "Spectral Magnitude Minimization".
    # A different one is "Time-Domain Pattern Search" or "Genetic Algorithm" or "Simulated Annealing".
    # But those are slow.
    
    # What if we use a **Randomized Coordinate Descent** with line search?
    # It is structurally different from the global spectral optimization.
    # It operates directly on the time-domain vector.
    
    def _line_search(x, idx, val_current, deadline):
        # Binary search for best step in x[idx]
        lo = -1.0
        hi = 1.0
        best_delta = 0.0
        best_val = val_current
        
        # Quick check if current is good
        for _ in range(20):
            if time.time() > deadline:
                break
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            
            x1 = x.copy()
            x1[idx] += m1
            v1 = _metric(x1)
            
            x2 = x.copy()
            x2[idx] += m2
            v2 = _metric(x2)
            
            if v1 < v2:
                hi = m2
            else:
                lo = m1
                
            if v1 < best_val:
                best_val = v1
                best_delta = m1
            if v2 < best_val:
                best_val = v2
                best_delta = m2
                
        return best_delta, best_val

    # Phase 1: Random Restart + Coordinate Descent
    # This is a local search method, distinct from the global spectral solver.
    
    # Start with a few random restarts to find a good basin
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        x_curr = rng.normal(0, 1, N)
        # Ensure non-zero sum
        if abs(np.sum(x_curr)) < 1e-3:
            x_curr[0] += 1.0
            
        val_curr = _metric(x_curr)
        
        # Coordinate Descent
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle indices to avoid bias
            indices = np.arange(N)
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5:
                    break
                delta, new_val = _line_search(x_curr, idx, val_curr, deadline - 5)
                if delta != 0.0 and new_val < val_curr:
                    x_curr[idx] += delta
                    val_curr = new_val
                    improved = True
        
        if val_curr < best_val:
            best_val = val_curr
            best_x = x_curr.copy()
            
    # Phase 2: Refinement with smaller steps
    # The line search above used a wide range [-1, 1].
    # Let's do a polish with smaller range.
    
    def _line_search_small(x, idx, val_current, deadline, scale=0.1):
        lo = -scale
        hi = scale
        best_delta = 0.0
        best_val = val_current
        
        for _ in range(15):
            if time.time() > deadline:
                break
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            
            x1 = x.copy()
            x1[idx] += m1
            v1 = _metric(x1)
            
            x2 = x.copy()
            x2[idx] += m2
            v2 = _metric(x2)
            
            if v1 < v2:
                hi = m2
            else:
                lo = m1
                
            if v1 < best_val:
                best_val = v1
                best_delta = m1
            if v2 < best_val:
                best_val = v2
                best_delta = m2
                
        return best_delta, best_val

    # Polish
    for sweep in range(10):
        if time.time() > deadline - 2:
            break
        indices = np.arange(N)
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline - 2:
                break
            delta, new_val = _line_search_small(best_x, idx, best_val, deadline - 2, scale=0.05)
            if delta != 0.0 and new_val < best_val:
                best_x[idx] += delta
                best_val = new_val

    return best_x

def construct():
    return _construct()
