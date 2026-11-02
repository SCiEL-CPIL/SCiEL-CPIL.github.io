import numpy as np
from scipy.optimize import minimize
import time

N = 50
_TIME_LIMIT = 140.0
_start_time = time.time()

def _score(h):
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _neg_score(h):
    return -_score(h)

def _nelder_mead_polish(h0, maxiter=2000):
    """Polish using Nelder-Mead, which handles the non-smooth max better."""
    h0 = np.array(h0, dtype=float)
    h0 = np.clip(h0, 0, None)
    if np.sum(h0) < 1e-12:
        return h0, 0.0
    h0 = h0 / np.sum(h0)
    
    def obj(x):
        x = np.clip(x, 0, None)
        s = np.sum(x)
        if s < 1e-12:
            return 0.0
        x = x / s
        return -_score(x)
    
    try:
        res = minimize(obj, h0, method='Nelder-Mead', 
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
        h_opt = np.clip(res.x, 0, None)
        if np.sum(h_opt) > 1e-12:
            h_opt = h_opt / np.sum(h_opt)
        return h_opt, _score(h_opt)
    except Exception:
        return h0, _score(h0)

def _coordinate_descent(h0, n_iters=500):
    """Coordinate descent with line search on each coordinate."""
    h = np.array(h0, dtype=float)
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-12:
        return h, 0.0
    h = h / np.sum(h)
    
    best_s = _score(h)
    
    for it in range(n_iters):
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        improved = False
        for i in range(N):
            # Try varying h[i] while keeping others fixed, then renormalize
            # Actually, since scale invariant, fix sum=1 by adjusting h[i] and one other
            # Simpler: try a grid of values for h[i], renormalize, check score
            
            old_val = h[i]
            # Try several multipliers
            for mult in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 8.0]:
                h_new = h.copy()
                h_new[i] = old_val * mult
                h_new = np.clip(h_new, 0, None)
                s = np.sum(h_new)
                if s < 1e-12:
                    continue
                h_new = h_new / s
                s_new = _score(h_new)
                if s_new > best_s + 1e-12:
                    h = h_new
                    best_s = s_new
                    improved = True
                    break
        if not improved:
            break
    
    return h, best_s

def construct():
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Diverse initializations - focus on structured patterns
    inits = []
    
    # 1. Two-point with varying ratio
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        h = np.zeros(N)
        h[0] = ratio
        h[1] = 1 - ratio
        inits.append(h)
    
    # 2. Three-point
    for a in [0.2, 0.3, 0.4]:
        for b in [0.2, 0.3]:
            c = 1 - a - b
            if c > 0:
                h = np.zeros(N)
                h[0] = a
                h[1] = b
                h[2] = c
                inits.append(h)
    
    # 3. Uniform
    inits.append(np.ones(N) / N)
    
    # 4. Sparse random
    rng = np.random.default_rng(42)
    for trial in range(20):
        h = np.zeros(N)
        k = rng.integers(2, 15)  # 2 to 14 nonzero
        idx = rng.choice(N, size=k, replace=False)
        vals = rng.dirichlet(np.ones(k))
        h[idx] = vals
        inits.append(h)
    
    # 5. Structured: alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0 / (N // 2 + N % 2)
    inits.append(h_alt)
    
    # 6. Structured: clustered at edges
    h_edge = np.zeros(N)
    h_edge[0] = 0.4
    h_edge[1] = 0.3
    h_edge[48] = 0.2
    h_edge[49] = 0.1
    inits.append(h_edge)
    
    # 7. Geometric decay
    h_geo = np.zeros(N)
    r = 0.95
    for i in range(N):
        h_geo[i] = r ** i
    h_geo /= np.sum(h_geo)
    inits.append(h_geo)
    
    # 8. Sinc-like (to flatten convolution)
    h_sinc = np.zeros(N)
    mid = (N - 1) / 2.0
    for i in range(N):
        x = (i - mid) / (N / 2)
        h_sinc[i] = np.sinc(x) if x != 0 else 1.0
    h_sinc = np.clip(h_sinc, 0, None)
    if np.sum(h_sinc) > 1e-12:
        h_sinc /= np.sum(h_sinc)
        inits.append(h_sinc)
    
    # Process all initializations
    for h0 in inits:
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        
        h0 = np.array(h0, dtype=float)
        h0 = np.clip(h0, 0, None)
        s0 = np.sum(h0)
        if s0 < 1e-12:
            continue
        h0 = h0 / s0
        
        # Polish with Nelder-Mead
        h_nm, s_nm = _nelder_mead_polish(h0, maxiter=500)
        
        # Then coordinate descent
        h_cd, s_cd = _coordinate_descent(h_nm, n_iters=200)
        
        # Final Nelder-Mead on the CD result
        h_final, s_final = _nelder_mead_polish(h_cd, maxiter=300)
        
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    # Additional random restarts if time permits
    rng2 = np.random.default_rng(123)
    for trial in range(50):
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        
        # Random sparse initialization
        h = np.zeros(N)
        k = rng2.integers(2, 20)
        idx = rng2.choice(N, size=k, replace=False)
        vals = rng2.dirichlet(np.ones(k) * rng2.uniform(0.5, 2.0))
        h[idx] = vals
        
        h_nm, s_nm = _nelder_mead_polish(h, maxiter=300)
        h_cd, s_cd = _coordinate_descent(h_nm, n_iters=100)
        
        if s_cd > best_score:
            best_score = s_cd
            best_h = h_cd
    
    return [float(x) for x in best_h]
