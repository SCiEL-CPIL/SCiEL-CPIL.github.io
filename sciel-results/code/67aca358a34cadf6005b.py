import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _smooth_max_conv(h, beta=100.0):
    """Smooth approximation of max(conv(h,h)) using log-sum-exp."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m <= 0:
        return 0.0
    # Use log-sum-exp for numerical stability
    shifted = beta * (conv - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / beta

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian - wider range of parameters
    for c in np.linspace(3, N-3, 15):
        for w in np.linspace(0.5, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(3, N-3, 15):
        for w in np.linspace(1, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # 4. Sparse peaks - try to minimize autoconvolution overlap
    # Use positions that are "spread out" to reduce convolution peaks
    for _ in range(200):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(3, 12)
        # Try to space peaks more evenly
        if num_peaks <= 3:
            indices = np.random.choice(N, num_peaks, replace=False)
        else:
            # Generate roughly evenly spaced positions with some randomness
            base_spacing = N / num_peaks
            positions = []
            for i in range(num_peaks):
                pos = int(i * base_spacing + np.random.uniform(-base_spacing*0.3, base_spacing*0.3))
                pos = max(0, min(N-1, pos))
                if pos not in positions:
                    positions.append(pos)
            indices = positions[:num_peaks]
        h_rand[indices] = 1.0
        # Add small noise to break symmetry
        h_rand += np.random.rand(N) * 0.05
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()
    
    # 5. Structured patterns - alternating, periodic
    for period in range(2, 10):
        if time.time() > deadline: break
        h_periodic = np.zeros(N)
        h_periodic[::period] = 1.0
        s = _score_raw(h_periodic)
        if s > best_score:
            best_score = s
            best_h = h_periodic.copy()
        
        # Shifted periodic
        for shift in range(period):
            h_shifted = np.zeros(N)
            h_shifted[shift::period] = 1.0
            s = _score_raw(h_shifted)
            if s > best_score:
                best_score = s
                best_h = h_shifted.copy()
    
    # 6. Two-peak configurations
    for i in range(N // 2):
        for j in range(i + N // 4, N):
            if time.time() > deadline: break
            h_two = np.zeros(N)
            h_two[i] = 1.0
            h_two[j] = 1.0
            s = _score_raw(h_two)
            if s > best_score:
                best_score = s
                best_h = h_two.copy()
    
    # Optimization using SLSQP with smooth max via log-sum-exp
    # Maximize L2^2 / max_conv subject to sum(h)=1, h>=0
    
    def neg_obj_smooth(h, beta=100.0):
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        smax = _smooth_max_conv(h, beta)
        if smax <= 0:
            return 0.0
        return -(l2_sq / smax)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    bounds = [(0.0, 1.0)] * N
    
    h_start = best_h / np.sum(best_h)
    
    # Try different beta values for smooth max - use higher values for better approximation
    for beta in [50.0, 100.0, 200.0, 500.0]:
        if time.time() > deadline: break
        def neg_obj_beta(h):
            return neg_obj_smooth(h, beta)
        try:
            res = minimize(neg_obj_beta, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 300, 'ftol': 1e-12})
            if res.success:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
                        h_start = h_candidate.copy()
        except Exception:
            pass
    
    # Generate diverse initializations
    inits = [best_h / np.sum(best_h), np.ones(N) / N]
    
    # More diverse Gaussians
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(0.5, N/2, 8):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # Triangular
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(1, N, 8):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # Sparse
    for _ in range(40):
        h_s = np.zeros(N)
        k = np.random.randint(3, 12)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # Smoothed random
    for _ in range(40):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        h_r /= np.sum(h_r)
        inits.append(h_r)
        
    # Periodic patterns
    for period in range(2, 8):
        h_p = np.zeros(N)
        h_p[::period] = 1.0
        h_p /= np.sum(h_p)
        inits.append(h_p)
    
    # Optimize from each init with smooth objective
    for h_init in inits:
        if time.time() > deadline: break
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        for beta in [100.0, 200.0]:
            if time.time() > deadline: break
            def neg_obj_b(h):
                return neg_obj_smooth(h, beta)
            try:
                res = minimize(neg_obj_b, h_init, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 150, 'ftol': 1e-10})
                if res.success:
                    h_candidate = np.maximum(0, res.x)
                    if np.sum(h_candidate) > 0:
                        h_candidate /= np.sum(h_candidate)
                        s_candidate = _score_raw(h_candidate)
                        if s_candidate > best_score:
                            best_score = s_candidate
                            best_h = h_candidate
            except Exception:
                pass

    # Final polish with exact (non-smooth) objective
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        
        def neg_obj_exact(h):
            l1 = np.sum(h)
            if l1 == 0: return 0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0
            return -(l2_sq / max_conv)
        
        # Multiple restarts from perturbed best
        for attempt in range(5):
            if time.time() > deadline - 2: break
            if attempt == 0:
                h_pert = h_start.copy()
            else:
                h_pert = best_h * (1 + 0.01 * attempt * np.random.randn(N))
                h_pert = np.maximum(0, h_pert)
                if np.sum(h_pert) > 0:
                    h_pert /= np.sum(h_pert)
                else:
                    continue
            
            try:
                res = minimize(neg_obj_exact, h_pert, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 500, 'ftol': 1e-14})
                if res.success:
                    h_candidate = np.maximum(0, res.x)
                    if np.sum(h_candidate) > 0:
                        h_candidate /= np.sum(h_candidate)
                        s_candidate = _score_raw(h_candidate)
                        if s_candidate > best_score:
                            best_score = s_candidate
                            best_h = h_candidate
                            h_start = h_candidate.copy()
            except Exception:
                pass

    return list(best_h)

# EVOLVE-BLOCK-END
