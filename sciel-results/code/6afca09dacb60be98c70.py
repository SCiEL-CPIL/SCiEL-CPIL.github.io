"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

Combined strategy:
1. Epigraph formulation via SLSQP (from inspiration 1) for structured optimization
2. Block-coordinate descent on symmetric pairs around dominant peak (from inspiration 1)
3. Sparse initialization with diverse patterns (from current program)
4. Local refinement with peak moving/scaling (from current program)
"""
import numpy as np
import time
import scipy.optimize as opt

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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse structured
    np.random.seed(42)
    for trial in range(40):
        k = np.random.randint(2, 12)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Concentrated Gaussians
    for c in np.linspace(3, N-3, 10):
        for w in np.linspace(0.5, N/3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(25):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 6. Structured sparse: pairs symmetric around center
    for k in range(2, 10):
        for offset in range(1, 8):
            h_pair = np.zeros(N)
            center = N // 2
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    h_pair[pos1] = 1.0
                if 0 <= pos2 < N:
                    h_pair[pos2] = 1.0
            if np.sum(h_pair) > 0:
                h_pair /= np.sum(h_pair)
                inits.append(h_pair)
    
    # 7. Alternating patterns
    for start in range(N):
        for gap in [2, 3, 4, 5, 7, 10]:
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                inits.append(h)
    
    # --- Phase 2: Epigraph Formulation Optimization ---
    num_conv = 2 * N - 1
    
    def objective(x):
        return x[N]
        
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
        
    def constraint_eq_fun(x):
        return np.sum(x[:N]) - 1.0
        
    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac
        
    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
        
    def constraint_ineq_jac(x):
        h = x[:N]
        jac = np.zeros((num_conv, N + 1))
        jac[:, N] = 1.0
        for j in range(N):
            k_start = j
            k_end = min(j + N, num_conv)
            if k_start < num_conv and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] = -2.0 * h[indices_h]
        return jac
        
    constraints = [
        {
            'type': 'eq',
            'fun': constraint_eq_fun,
            'jac': constraint_eq_jac
        },
        {
            'type': 'ineq',
            'fun': constraint_ineq_fun,
            'jac': constraint_ineq_jac
        }
    ]
    
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]
    
    def run_epigraph(h_init, maxiter=200, ftol=1e-10):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
        x0 = np.concatenate((h_init, [M_init]))
        
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        return None
    
    # Run epigraph from multiple inits (limit to keep within time budget)
    max_epigraph_runs = min(len(inits), 30)
    for i, h_init in enumerate(inits[:max_epigraph_runs]):
        if time.time() > deadline - 40:
            break
        h_candidate = run_epigraph(h_init, maxiter=200, ftol=1e-10)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 3: Block-Coordinate Descent Refinement ---
    def block_coordinate_refine(h_start, max_iter=30):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 10:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            
            # For each position i, try to optimize h[i] and h[k_star - i] (if valid)
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                    
                hi = h[i]
                hj = h[j]
                
                t_min = -hi
                t_max = hj
                
                if t_max - t_min < 1e-15:
                    continue
                    
                best_t = 0.0
                best_local_score = _score_raw(h)
                
                for t in np.linspace(t_min * 0.8, t_max * 0.8, 15):
                    h_test = h.copy()
                    h_test[i] += t
                    h_test[j] -= t
                    if np.any(h_test < -1e-15):
                        continue
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > best_local_score:
                        best_local_score = s_test
                        best_t = t
                
                if abs(best_t) > 1e-10:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    if np.sum(h) < 1e-15:
                        break
                    h /= np.sum(h)
                    improved = True
            
            # Also try single-coordinate scaling
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                if h[i] < 1e-10:
                    continue
                orig = h[i]
                for factor in [0.5, 0.7, 1.3, 1.5, 2.0]:
                    h_test = h.copy()
                    h_test[i] = orig * factor
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > best_local_score:
                        h = h_test.copy()
                        improved = True
                        break
        
        return h
    
    # Apply block-coordinate refinement to best found so far
    if best_score > 0:
        h_refined = block_coordinate_refine(best_h, max_iter=30)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # --- Phase 4: Local refinement with peak moving/scaling (from current program) ---
    def local_refine(h_start, max_iter=100):
        h_curr = h_start.copy()
        h_curr = np.clip(h_curr, 0, None)
        if np.sum(h_curr) < 1e-15:
            return h_curr
        h_curr /= np.sum(h_curr)
        curr_score = _score_raw(h_curr)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            nonzero_idx = np.where(h_curr > 1e-6)[0]
            
            for idx in nonzero_idx:
                if time.time() > deadline - 5:
                    break
                
                orig_val = h_curr[idx]
                
                # Try scaling up and down
                for factor in [0.5, 0.7, 1.3, 1.5, 2.0]:
                    h_try = h_curr.copy()
                    h_try[idx] = orig_val * factor
                    h_try = np.clip(h_try, 0, None)
                    if np.sum(h_try) < 1e-15:
                        continue
                    h_try /= np.sum(h_try)
                    s = _score_raw(h_try)
                    if s > curr_score:
                        curr_score = s
                        h_curr = h_try.copy()
                        improved = True
                        break
                
                # Try moving the peak
                for offset in [-2, -1, 1, 2]:
                    new_idx = idx + offset
                    if 0 <= new_idx < N and h_curr[new_idx] < 1e-6:
                        h_try = h_curr.copy()
                        h_try[new_idx] = h_try[idx]
                        h_try[idx] = 0.0
                        h_try = np.clip(h_try, 0, None)
                        if np.sum(h_try) < 1e-15:
                            continue
                        h_try /= np.sum(h_try)
                        s = _score_raw(h_try)
                        if s > curr_score:
                            curr_score = s
                            h_curr = h_try.copy()
                            improved = True
                            break
            
            # Try adding a new peak
            if not improved:
                zero_idx = np.where(h_curr < 1e-6)[0]
                if len(zero_idx) > 0:
                    new_idx = np.random.choice(zero_idx)
                    h_try = h_curr.copy()
                    h_try[new_idx] = 0.5
                    h_try = np.clip(h_try, 0, None)
                    if np.sum(h_try) < 1e-15:
                        continue
                    h_try /= np.sum(h_try)
                    s = _score_raw(h_try)
                    if s > curr_score:
                        curr_score = s
                        h_curr = h_try.copy()
            
            # Try removing a peak
            if not improved and len(nonzero_idx) > 1:
                remove_idx = np.random.choice(nonzero_idx)
                h_try = h_curr.copy()
                h_try[remove_idx] = 0.0
                h_try = np.clip(h_try, 0, None)
                if np.sum(h_try) < 1e-15:
                    continue
                h_try /= np.sum(h_try)
                s = _score_raw(h_try)
                if s > curr_score:
                    curr_score = s
                    h_curr = h_try.copy()
            
            if not improved:
                break
        
        return h_curr
    
    h_local = local_refine(best_h, max_iter=100)
    s_local = _score_raw(h_local)
    if s_local > best_score:
        best_score = s_local
        best_h = h_local.copy()
    
    # Return best found
    if best_h is not None:
        return list(best_h)
    return list(np.ones(N))
