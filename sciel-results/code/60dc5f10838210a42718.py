import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA = 1.0 / N

def autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h."""
    n = len(h)
    k_vals = np.arange(-(n-1), n)
    ff_vals = np.zeros(len(k_vals))
    
    for idx, k in enumerate(k_vals):
        total = 0.0
        for i in range(n):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[idx] = total * DELTA
    
    return np.max(ff_vals)

def ratio(h):
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    best_h = np.ones(N)
    best_score = ratio(best_h)
    
    np.random.seed(42)
    
    # Use Nelder-Mead which is much faster than SLSQP
    # and doesn't require gradient computation
    
    initializations = []
    
    # Structured initializations
    t = np.linspace(-1, 1, N)
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Various smooth profiles
    initializations.append(np.maximum(1 - t**2, 0.01))
    initializations.append(np.maximum(t**2 + 0.1, 0.01))
    initializations.append(np.maximum(1 + 0.8*np.cos(np.pi*t), 0.05))
    initializations.append(np.exp(-t**2/0.2))
    initializations.append(np.maximum(1 + np.sin(2*np.pi*t), 0.05))
    initializations.append(np.maximum(1 + np.sin(4*np.pi*t), 0.05))
    
    # Sparse patterns
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    h_sp4 = np.zeros(N)
    h_sp4[::4] = 1.0
    initializations.append(h_sp4)
    
    # Random initializations
    for _ in range(20):
        h_r = np.random.uniform(0, 1, N)
        # Make some sparse
        if np.random.random() < 0.5:
            h_r[np.random.random(N) > 0.3] = 0
        initializations.append(h_r)
    
    # Bimodal
    for center in [-0.5, 0.5]:
        initializations.append(np.maximum(1.0 + 1.5*np.exp(-((t-center)**2)/0.08), 0.05))
    
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 3:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        def neg_ratio(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 0.0
            return -ratio(x / s)
        
        try:
            # Nelder-Mead is much faster
            result = minimize(neg_ratio, h0.copy(), method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            
            if time.time() - start_time > DEADLINE - 3:
                break
            
            if result.fun < -best_score + 1e-10:
                h_candidate = np.maximum(result.x, 0)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate = h_candidate / s
                    sc = ratio(h_candidate)
                    if sc > best_score:
                        best_score = sc
                        best_h = h_candidate.copy()
        except Exception:
            continue
    
    return best_h.tolist()
