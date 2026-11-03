import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
DX = (1/4 - (-1/4)) / N  # spacing = 0.02... wait, 50 intervals on [-1/4,1/4]?

# "50 nonnegative heights on equal intervals of [-1/4,1/4]"
# This likely means 50 points dividing [-1/4,1/4] into 50 equal intervals, so 51 points? 
# Or 50 intervals meaning 50 subintervals, so the heights correspond to 50 subintervals.
# Let's assume 50 heights, each representing a subinterval of width dx = (1/2)/50 = 0.01

DX = 0.01  # 1/100

def compute_ratio(h):
    """h is array of N nonnegative heights. Returns the ratio."""
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h) * DX
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2) * DX
    # Autoconvolution
    hh = np.convolve(h, h, mode='full') * DX
    sup_norm = np.max(hh)
    if sup_norm == 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def objective(h):
    """Minimize negative ratio."""
    return -compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Start with a delta-like initial guess
    h0 = np.zeros(N)
    h0[0] = 1.0
    
    best_h = h0.copy()
    best_val = compute_ratio(best_h)
    
    # Try multiple initializations with differential evolution or Nelder-Mead
    from scipy.optimize import minimize
    
    # Constraint: sum(h) = 1 (scale invariance), h >= 0
    # Use penalty method for non-negativity
    
    def neg_ratio_constrained(x):
        h = np.maximum(x, 0)  # project to non-negative
        # Normalize to sum = 1
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h = h / s
        return -compute_ratio(h)
    
    # Run several Nelder-Mead optimizations from different starts
    starts = []
    
    # Delta at various positions
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        starts.append(h)
    
    # Uniform
    h = np.ones(N) / N
    starts.append(h)
    
    # Two-point spreads
    for i in range(0, N, 5):
        for j in range(i+1, N, 5):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            starts.append(h)
    
    # Random starts
    np.random.seed(42)
    for _ in range(20):
        h = np.random.random(N)
        h = h / np.sum(h)
        starts.append(h)
    
    # Evaluate all starts
    for h in starts:
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        if time.time() > deadline:
            break
    
    # Try local optimization from best few starts
    top_starts = sorted(starts, key=lambda h: -compute_ratio(h))[:10]
    
    for h_start in top_starts:
        if time.time() > deadline:
            break
        try:
            result = minimize(neg_ratio_constrained, h_start, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < -best_val:
                h_opt = np.maximum(result.x, 0)
                s = np.sum(h_opt)
                if s > 1e-10:
                    h_opt = h_opt / s
                    val = compute_ratio(h_opt)
                    if val > best_val:
                        best_val = val
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Convert to list of floats
    return list(best_h)
