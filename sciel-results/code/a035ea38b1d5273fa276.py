"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 130.0  # safety margin before 150s kill

def _evaluate(h):
    """Compute the value metric for a given height vector."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h, mode='full')
    m = len(g)
    w = 1.0 / N
    y = np.zeros(m + 2)
    y[1:-1] = g
    # l2 = (w/3) * sum(y[i]^2 + y[i]*y[i+1] + y[i+1]^2) for i in 0..m
    y2 = y**2
    cross = y * np.roll(y, -1)
    # sum over i=0..m of (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # = sum(y[i]^2 for i=0..m) + sum(y[i]*y[i+1] for i=0..m) + sum(y[i+1]^2 for i=0..m)
    # = sum(y[0..m]^2) + sum(y[0..m]*y[1..m+1]) + sum(y[1..m+1]^2)
    l2 = (w / 3.0) * (np.sum(y2[:m+1]) + np.sum(y[:m+1] * y[1:m+2]) + np.sum(y2[1:m+2]))
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-15:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _tv_project(h, lam, max_iter=50):
    """
    1D total variation denoising via iterative soft-thresholding (Rudin-Otsu style).
    Projects h onto the set of piecewise-constant-like signals by penalizing TV.
    Uses a simple iterative approach: h_{k+1} = soft_threshold(h_k - lam * grad, lam)
    where grad is the discrete gradient.
    """
    h = np.array(h, dtype=np.float64)
    n = len(h)
    for _ in range(max_iter):
        # Compute discrete gradient (forward difference)
        grad = np.zeros(n)
        grad[1:] = h[1:] - h[:-1]
        # Soft threshold the gradient
        grad_st = np.sign(grad) * np.maximum(np.abs(grad) - lam, 0.0)
        # Update: h_new = h - grad_st (this is a simplified RO step)
        h_new = h - grad_st
        # Enforce nonnegativity
        h_new = np.maximum(h_new, 0.0)
        if np.max(np.abs(h_new - h)) < 1e-10:
            h = h_new
            break
        h = h_new
    return h

def _block_init(pattern_type, seed_val=1.0):
    """Create structured block-like initializations."""
    h = np.zeros(N)
    if pattern_type == 0:
        # Single central block
        h[15:35] = seed_val
    elif pattern_type == 1:
        # Two symmetric blocks
        h[10:20] = seed_val
        h[30:40] = seed_val
    elif pattern_type == 2:
        # Three blocks
        h[5:15] = seed_val
        h[20:30] = seed_val
        h[35:45] = seed_val
    elif pattern_type == 3:
        # Wide central block
        h[10:40] = seed_val
    elif pattern_type == 4:
        # Narrow central spike
        h[23:27] = seed_val
    elif pattern_type == 5:
        # Alternating blocks
        h[0:10] = seed_val
        h[20:30] = seed_val
        h[40:50] = seed_val
    elif pattern_type == 6:
        # Central peak with tails
        h[20:30] = seed_val
        h[15:20] = 0.5 * seed_val
        h[30:35] = 0.5 * seed_val
    elif pattern_type == 7:
        # Symmetric U-shape
        h[0:15] = seed_val
        h[35:50] = seed_val
    elif pattern_type == 8:
        # Gaussian-like (but we'll TV-project it)
        x = np.linspace(-1, 1, N)
        h = np.exp(-2.0 * x**2)
        h *= seed_val / max(h.max(), 1e-10)
    elif pattern_type == 9:
        # Triangle
        h = np.abs(1.0 - 2.0 * np.linspace(0, 1, N))
        h *= seed_val / max(h.max(), 1e-10)
    return h

def _coordinate_ascent(h, max_iter=200, tol=1e-10):
    """Coordinate ascent: optimize one variable at a time."""
    h = np.array(h, dtype=np.float64)
    best_val = _evaluate(h)
    for it in range(max_iter):
        improved = False
        for i in range(N):
            old = h[i]
            # Try a few perturbations around current value
            for delta in [-0.1, -0.01, 0.01, 0.1, -0.5, 0.5, 1.0]:
                h[i] = max(0.0, old + delta)
                v = _evaluate(h)
                if v > best_val + tol:
                    best_val = v
                    improved = True
            # Also try setting to 0
            h[i] = 0.0
            v = _evaluate(h)
            if v > best_val + tol:
                best_val = v
                improved = True
            else:
                h[i] = old
        if not improved:
            break
    return h, best_val

def _nelder_mead(h, max_iter=100):
    """Nelder-Mead optimization."""
    h = np.array(h, dtype=np.float64)
    
    def neg_value(x):
        x = np.maximum(x, 0.0)
        return -_evaluate(x)
    
    try:
        res = minimize(neg_value, h, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        h_opt = np.maximum(res.x, 0.0)
        return h_opt, -res.fun
    except:
        return h, _evaluate(h)

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Try multiple structured initializations
    for pattern in range(10):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        h = _block_init(pattern)
        val = _evaluate(h)
        
        # TV projection to encourage piecewise-constant structure
        for lam in [0.01, 0.05, 0.1]:
            if time.time() - start_time > DEADLINE - 5:
                break
            h_tv = _tv_project(h, lam, max_iter=30)
            val_tv = _evaluate(h_tv)
            if val_tv > val:
                h = h_tv
                val = val_tv
        
        if val > best_val:
            best_val = val
            best_h = h.copy()
        
        # Coordinate ascent from this initialization
        if time.time() - start_time < DEADLINE - 10:
            h_ca, val_ca = _coordinate_ascent(h, max_iter=100)
            if val_ca > best_val:
                best_val = val_ca
                best_h = h_ca.copy()
        
        # Nelder-Mead from best so far
        if time.time() - start_time < DEADLINE - 15:
            h_nm, val_nm = _nelder_mead(best_h, max_iter=50)
            if val_nm > best_val:
                best_val = val_nm
                best_h = h_nm.copy()
    
    # Final coordinate ascent polish
    if time.time() - start_time < DEADLINE - 5:
        h_final, val_final = _coordinate_ascent(best_h, max_iter=50)
        if val_final > best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    # Final TV projection polish
    if time.time() - start_time < DEADLINE - 2:
        for lam in [0.005, 0.02]:
            h_tv = _tv_project(best_h, lam, max_iter=20)
            val_tv = _evaluate(h_tv)
            if val_tv > best_val:
                best_val = val_tv
                best_h = h_tv.copy()
    
    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
