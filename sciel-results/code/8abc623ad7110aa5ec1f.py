import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the ratio: L2^2 / (L1 * Sup(AutoConv))
    where AutoConv is the continuous autoconvolution of the step function.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Parametric initializations ---
    
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # 4. Random sparse initialization
    for _ in range(50):
        if time.time() > deadline - 20: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # --- Phase 2: Optimization ---
    
    # We optimize h with sum(h)=1 to find the best shape.
    # The ratio V(h) = L2^2 / (L1 * Sup).
    # If L1=1, V(h) = L2^2 / Sup.
    # Note: Sup = DELTA * max(conv(h,h)).
    # So we maximize L2^2 / max(conv(h,h)).
    
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 == 0: return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0: return 0
        return -(l2_sq / max_conv)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Start from best_h (normalized)
    h_start = best_h / np.sum(best_h)
    
    # Try SLSQP
    try:
        res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success:
            h_candidate = res.x
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
    except Exception:
        pass

    # Generate diverse initializations for restarts
    inits = [h_start]
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    # Gaussians
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/2, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # Triangles
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # Sparse (Sidon-like)
    for _ in range(20):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline - 10: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(neg_obj_h, h_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # --- Phase 3: Refinement with Smooth Surrogate ---
    # The objective L2^2 / max(conv) is non-smooth due to max.
    # We can use a smooth approximation for the max term: soft-max.
    # soft_max(x, t) = t * log(sum(exp(x_i / t)))
    # As t -> 0, soft_max -> max(x).
    # We maximize L2^2 / soft_max(conv, t).
    # This is equivalent to minimizing -log(L2^2) + log(soft_max(conv, t)).
    # Or simply minimizing -L2^2 / soft_max(conv, t).
    
    def neg_obj_smooth(h, t):
        l1 = np.sum(h)
        if l1 == 0: return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        
        # Soft max
        # Scale h_conv to avoid overflow/underflow in exp
        m = np.max(h_conv)
        exp_conv = np.exp(h_conv / t - m/t)
        soft_max_val = t * (m/t + np.log(np.sum(exp_conv)))
        
        if soft_max_val <= 0: return 0
        return -(l2_sq / soft_max_val)
    
    # Try different temperature t
    for t in [0.01, 0.05, 0.1, 0.5]:
        if time.time() > deadline - 5: break
        
        h_start = best_h / np.sum(best_h)
        
        try:
            res = minimize(lambda h: neg_obj_smooth(h, t), h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                h_candidate = res.x
                # Evaluate with the TRUE non-smooth score
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Ensure nonnegativity and normalize
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) > 1e-10:
        best_h = best_h / np.sum(best_h)

    return list(best_h)
