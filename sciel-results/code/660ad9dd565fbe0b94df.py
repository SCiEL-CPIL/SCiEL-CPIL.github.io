import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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
    deadline = 145.0  # Safety margin before 150s kill
    
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
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(0.5, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(0.5, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # 4. Edge-concentrated (high at boundaries)
    for alpha in np.linspace(0.5, 3.0, 10):
        h_edge = np.exp(-alpha * (x_idx - N/2)**2 / N)
        s = _score_raw(h_edge)
        if s > best_score:
            best_score = s
            best_h = h_edge.copy()
    
    # 5. Alternating patterns
    for period in [2, 3, 4, 5]:
        for phase in range(period):
            h_alt = np.where((x_idx - phase) % period < period//2, 1.0, 0.0)
            if np.sum(h_alt) > 0:
                s = _score_raw(h_alt)
                if s > best_score:
                    best_score = s
                    best_h = h_alt.copy()

    # 6. Random sparse initialization
    for _ in range(150):
        if time.time() > deadline - 25: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(3, 20)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = np.random.uniform(0.5, 1.0, num_peaks)
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
    
    def neg_obj_smooth(h, t):
        """Smooth approximation using log-sum-exp for the max."""
        l1 = np.sum(h)
        if l1 == 0: return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        
        # Soft max with temperature t
        m = np.max(h_conv)
        if m <= 0: return 0
        exp_conv = np.exp(h_conv / t - m/t)
        soft_max_val = t * (m/t + np.log(np.sum(exp_conv)))
        
        if soft_max_val <= 0: return 0
        return -(l2_sq / soft_max_val)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Start from best_h (normalized)
    h_start = best_h / np.sum(best_h)
    
    # Try SLSQP with non-smooth objective
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
    
    # Try SLSQP with smooth surrogate at various temperatures
    for t in [0.005, 0.01, 0.02, 0.05, 0.1]:
        if time.time() > deadline - 30: break
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(lambda h: neg_obj_smooth(h, t), h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 800, 'ftol': 1e-13})
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
    for c in np.linspace(2, N-2, 8):
        for w in np.linspace(0.5, N/2, 8):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # Triangles
    for c in np.linspace(2, N-2, 8):
        for w in np.linspace(0.5, N, 8):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # Sparse (Sidon-like)
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(3, 20)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = np.random.uniform(0.3, 1.0, k)
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # Random smooth
    for _ in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)
    
    # Edge-concentrated
    for alpha in np.linspace(0.5, 3.0, 5):
        h_edge = np.exp(-alpha * (x_idx - N/2)**2 / N)
        h_edge /= np.sum(h_edge)
        inits.append(h_edge)

    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline - 15: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(neg_obj_h, h_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300, 'ftol': 1e-11})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # --- Phase 3: Coordinate Descent Refinement ---
    # After gradient-based optimization, do coordinate-wise refinement
    # to handle the non-smooth max more precisely.
    
    def coordinate_descent(h, max_iters=200):
        """Iteratively adjust each coordinate to maximize the objective."""
        h = h.copy()
        for iteration in range(max_iters):
            improved = False
            for i in range(N):
                old_val = h[i]
                # Try a grid of values for coordinate i
                candidates = np.unique(np.concatenate([
                    [0.0],
                    np.linspace(0, max(old_val * 2, 0.01), 10),
                    [old_val]
                ]))
                best_cand = old_val
                best_score_i = _score_raw(h)
                
                for cand in candidates:
                    h[i] = cand
                    s_i = _score_raw(h)
                    if s_i > best_score_i:
                        best_score_i = s_i
                        best_cand = cand
                h[i] = best_cand
                if best_cand != old_val:
                    improved = True
            if not improved:
                break
        return h
    
    if time.time() < deadline - 10:
        h_cd = coordinate_descent(best_h, max_iters=50)
        s_cd = _score_raw(h_cd)
        if s_cd > best_score:
            best_score = s_cd
            best_h = h_cd

    # --- Phase 4: Final polish with tight tolerance ---
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-15})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
    
    # Final coordinate descent polish
    if time.time() < deadline - 3:
        h_cd = coordinate_descent(best_h, max_iters=30)
        s_cd = _score_raw(h_cd)
        if s_cd > best_score:
            best_score = s_cd
            best_h = h_cd

    return list(best_h)
