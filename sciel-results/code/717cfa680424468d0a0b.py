import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the score: L2^2 / (L1 * sup(autoconv))."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Gaussian peaks
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            inits.append(h_gauss)
            
    # 3. Triangular peaks
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                inits.append(h_tri)
                
    # 4. Sparse peaks (Sidon-like intuition)
    for _ in range(100):
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        inits.append(h_rand)
        
    # 5. Two-peak patterns
    for c1 in np.linspace(5, N/2, 5):
        for c2 in np.linspace(N/2, N-5, 5):
            for w in [2, 3, 5, 8]:
                h_two = np.exp(-0.5 * ((x_idx - c1) / w)**2) + np.exp(-0.5 * ((x_idx - c2) / w)**2)
                inits.append(h_two)
                
    # 6. Random smooth patterns
    for _ in range(50):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        inits.append(h_r)
        
    # 7. Ramp patterns
    for direction in [1, -1]:
        h_ramp = np.arange(N) if direction == 1 else np.arange(N)[::-1]
        inits.append(h_ramp.astype(float))
        
    # 8. Chebyshev-like patterns
    for k in [1, 2, 3, 5]:
        h_cheb = np.abs(np.cos(k * np.pi * x_idx / (N-1)))
        inits.append(h_cheb)
        
    # 9. Fourier-like patterns
    for freq in [1, 2, 3, 5, 8]:
        h_fourier = np.abs(np.sin(freq * np.pi * x_idx / N)) ** 2
        inits.append(h_fourier)
        
    # 10. Alternating patterns
    for period in [2, 3, 4, 5]:
        h_alt = np.where((x_idx % period) < period//2, 1.0, 0.1)
        inits.append(h_alt)
        
    # Score all initializations
    for h_init in inits:
        if time.time() > deadline:
            break
        s = _score_raw(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()
            
    # Now optimize from best initializations
    # Use SLSQP on h directly with constraints: sum(h) = 1, h >= 0
    # Objective: maximize L2^2 / max(conv(h,h)) which is equivalent to maximizing score when L1=1
    
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0.0
        return -(l2_sq / max_conv)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Normalize best_h for optimization
    h_start = best_h / np.sum(best_h)
    
    # First optimization pass
    try:
        res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
        if res.success:
            s_candidate = _score_raw(res.x)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = res.x.copy()
    except Exception:
        pass
    
    # Generate more diverse starting points for optimization
    opt_inits = [h_start]
    
    # Add top-scoring initializations
    scored_inits = []
    for h_init in inits:
        if time.time() > deadline:
            break
        s = _score_raw(h_init)
        scored_inits.append((s, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    for _, h_init in scored_inits[:20]:
        opt_inits.append(h_init / np.sum(h_init))
        
    # Optimize from each starting point
    for h_opt_init in opt_inits:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(neg_obj_h, h_opt_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 300, 'ftol': 1e-9})
            if res.success:
                s_candidate = _score_raw(res.x)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = res.x.copy()
        except Exception:
            pass
            
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                s_candidate = _score_raw(res.x)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Ensure non-negativity
    best_h = np.maximum(0, best_h)
    
    return list(best_h)
