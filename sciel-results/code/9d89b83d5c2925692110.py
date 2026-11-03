import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Normalize to L1=1 for the ratio calculation
    # The score is scale invariant if we consider the ratio R = L2^2 / (L1 * Sup).
    # R(c*h) = R(h)/c.
    # If the evaluator normalizes to L1=1, then R(h) = L2^2 / Sup.
    # Let's assume the evaluator normalizes to L1=1.
    # So we maximize L2^2 / Sup(h*h) subject to L1=1.
    
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return l2_sq / autoconv_max

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Linear ramps
    for slope in [1, -1]:
        h_lin = np.linspace(1, N, N) if slope > 0 else np.linspace(N, 1, N)
        inits.append(h_lin)
        
    # 3. Gaussians
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            inits.append(h_g)
            
    # 4. Triangles
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                inits.append(h_t)
                
    # 5. Random sparse
    for _ in range(50):
        h_r = np.zeros(N)
        k = np.random.randint(5, 20)
        idx = np.random.choice(N, k, replace=False)
        h_r[idx] = 1.0
        inits.append(h_r)
        
    # 6. Random smooth
    for _ in range(50):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        inits.append(h_r)

    def optimize_from(h_init):
        h_init = np.asarray(h_init, dtype=np.float64)
        if np.sum(h_init) == 0:
            return h_init, 0.0
        
        h_curr = h_init / np.sum(h_init)
        score_curr = _score_raw(h_curr)
        
        # Iterative peak index optimization
        for _ in range(10):
            h_conv = np.convolve(h_curr, h_curr, mode='full')
            k_star = np.argmax(h_conv)
            
            # Objective: maximize sum(h^2) / sum(h_i h_{k_star-i})
            # Note: h is constrained to sum=1, h>=0
            
            def neg_obj(h):
                h = np.maximum(h, 0)
                if np.sum(h) == 0: return 0
                h_n = h / np.sum(h)
                l2_sq = np.sum(h_n**2)
                # Compute specific convolution term
                # c_k = sum_i h_i h_{k-i}
                # We need to be careful with indices.
                # np.convolve(h, h) gives c_0 ... c_{2N-2}
                # c_k = sum_{i=0}^{N-1} h_i h_{k-i} (with h_j=0 if j<0 or j>=N)
                # So term = sum_{i=max(0, k-N+1)}^{min(N-1, k)} h_i h_{k-i}
                
                # Efficient calculation:
                # Use np.dot on appropriate slices
                i_start = max(0, k_star - (N-1))
                i_end = min(N, k_star + 1)
                i_vals = np.arange(i_start, i_end)
                j_vals = k_star - i_vals
                # h_i * h_j
                term = np.sum(h_n[i_vals] * h_n[j_vals])
                
                if term == 0: return 0
                return -(l2_sq / term)
            
            constraints = [
                {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
            ]
            bounds = [(0.0, 1.0)] * N
            
            try:
                res = minimize(neg_obj, h_curr, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
                if res.success:
                    h_new = res.x
                    h_new = np.maximum(h_new, 0)
                    h_new = h_new / np.sum(h_new)
                    score_new = _score_raw(h_new)
                    if score_new > score_curr:
                        h_curr = h_new
                        score_curr = score_new
                    # Check if peak index changed
                    h_conv_new = np.convolve(h_curr, h_curr, mode='full')
                    k_new = np.argmax(h_conv_new)
                    if k_new == k_star:
                        break
            except Exception:
                pass
        
        return h_curr, score_curr

    for h_init in inits:
        if time.time() > deadline: break
        h_opt, s_opt = optimize_from(h_init)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt

    # Final polish
    if time.time() < deadline - 5:
        h_opt, s_opt = optimize_from(best_h)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt

    return list(best_h)
