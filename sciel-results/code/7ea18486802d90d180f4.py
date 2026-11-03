import numpy as np
import time
from scipy.optimize import minimize

# Constants
N = 50
DELTA = 0.5 / N
_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio_discrete(h):
    """
    Compute the ratio R = sum(h^2) / (sum(h) * max(conv(h,h))).
    This is the likely metric if the evaluator uses discrete sums.
    """
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def _optimize_epigraph(h_init, max_iter=500, ftol=1e-8):
    """
    Optimize h to minimize the peak of the autoconvolution, 
    subject to sum(h)=1 and h>=0.
    Variables: x = [h_0, ..., h_{N-1}, M]
    Objective: Minimize M
    Constraints:
      1. sum(h) - 1 = 0
      2. M - (h*h)_k >= 0 for all k
    """
    def obj(x):
        return x[N]

    def obj_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad

    def constraint_sum(x):
        return np.sum(x[:N]) - 1.0

    def constraint_sum_grad(x):
        grad = np.zeros(N + 1)
        grad[:N] = 1.0
        return grad

    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        h_clip = np.clip(h, 0, None)
        c = np.convolve(h_clip, h_clip, mode="full")
        return M - c

    def constraints_autoconv_grad(x):
        h = x[:N]
        grad = np.zeros((2 * N - 1, N + 1))
        for k in range(2 * N - 1):
            derivs = np.zeros(N)
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    derivs[i] = 2.0 * h[j]
            grad[k, :N] = -derivs
            grad[k, N] = 1.0
        return grad

    h_init = np.asarray(h_init, dtype=np.float64)
    s = np.sum(h_init)
    if s <= 0:
        h_init = np.ones(N) / N
        s = 1.0
    h_init = h_init / s
    
    c_init = np.convolve(h_init, h_init, mode="full")
    M_init = np.max(c_init)
    x0 = np.concatenate([h_init, [M_init]])
    
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]
    
    try:
        res = minimize(
            obj, 
            x0, 
            method='SLSQP', 
            jac=obj_grad,
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': ftol}
        )
        
        if res.success or res.fun < M_init:
            h_res = res.x[:N]
            h_res = np.clip(h_res, 0, None)
            s = np.sum(h_res)
            if s > 0:
                h_res = h_res / s
            else:
                return None, 0.0
                
            ratio_res = _ratio_discrete(h_res)
            return h_res, ratio_res
    except Exception:
        pass
    return None, 0.0

def _coordinate_descent_refinement(h, steps=20):
    """
    Perform coordinate descent to maximize the ratio.
    For each coordinate i, find the optimal value h_i that maximizes the ratio,
    given the other coordinates fixed.
    """
    h = h.copy()
    current_ratio = _ratio_discrete(h)
    
    for _ in range(steps):
        improved = False
        for i in range(N):
            # Save current value
            h_i_old = h[i]
            
            # We want to maximize R(h) = (sum h^2) / (sum h * max conv)
            # This is complex to solve analytically for h_i.
            # Instead, we can try a small grid search or gradient step.
            # Let's try a simple line search around h_i.
            
            # If h_i is 0, try small positive values
            # If h_i > 0, try scaling
            
            # Simple approach: try 10 perturbations
            best_h_i = h_i_old
            best_r = current_ratio
            
            # Perturbations: scale by 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, and 0
            scales = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 4.0]
            for scale in scales:
                h_test = h.copy()
                h_test[i] = h_i_old * scale
                # Renormalize to sum=1? No, the ratio is scale invariant in the discrete sense?
                # R(c h) = R(h). So we don't need to renormalize for comparison, 
                # but we should keep the sum close to 1 for consistency with the optimizer.
                # Actually, R(h) is scale invariant. So we can just evaluate R(h_test).
                r_test = _ratio_discrete(h_test)
                if r_test > best_r:
                    best_r = r_test
                    best_h_i = h_test[i]
            
            if best_h_i != h_i_old:
                h[i] = best_h_i
                current_ratio = best_r
                improved = True
        
        if not improved:
            break
            
    # Final normalization
    s = np.sum(h)
    if s > 0:
        h /= s
    
    return h, _ratio_discrete(h)

def _find_sidon_support(target_size, N):
    """
    Greedy construction of a Sidon set.
    """
    if target_size <= 0:
        return []
    
    support = [0]
    used_sums = {0}
    
    for _ in range(target_size - 1):
        best_candidate = None
        best_score = -1
        
        for candidate in range(N):
            if candidate in support:
                continue
            
            collisions = 0
            for s in support:
                if candidate + s in used_sums:
                    collisions += 1
            
            score = -collisions
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        if best_candidate is not None:
            support.append(best_candidate)
            for s in support:
                used_sums.add(best_candidate + s)
        
        if best_candidate is None:
            break
            
    return support

def construct():
    np.random.seed(12345)
    
    best_h = None
    best_ratio = -np.inf

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = _optimize_epigraph(h_uniform, max_iter=300)
    if h_opt is not None and s_opt > best_ratio:
        best_ratio = s_opt
        best_h = h_opt.copy()

    # 2. Try Sidon-like structured supports
    for target_k in [5, 6, 7, 8, 10, 12]:
        if _time_left() < 15.0:
            break
        
        support = _find_sidon_support(target_k, N)
        if len(support) >= 3:
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0 / len(support)
            
            h_opt, s_opt = _optimize_epigraph(h_sidon, max_iter=300)
            if h_opt is not None and s_opt > best_ratio:
                best_ratio = s_opt
                best_h = h_opt.copy()
            
            # Refine with coordinate descent
            h_ref, s_ref = _coordinate_descent_refinement(h_opt, steps=10)
            if s_ref > best_ratio:
                best_ratio = s_ref
                best_h = h_ref

    # 3. Try random sparse starts
    num_restarts = 20
    for i in range(num_restarts):
        if _time_left() < 10.0:
            break
        
        K = np.random.randint(3, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = _optimize_epigraph(h_rand, max_iter=200)
        if h_opt is not None and s_opt > best_ratio:
            best_ratio = s_opt
            best_h = h_opt.copy()
            
        # Refine
        h_ref, s_ref = _coordinate_descent_refinement(h_opt, steps=10)
        if s_ref > best_ratio:
            best_ratio = s_ref
            best_h = h_ref

    # 4. Polish the best solution
    if best_h is not None and _time_left() > 20.0:
        h_polish_init = best_h.copy()
        h_polish_init[h_polish_init < 1e-6] = 0.0
        s = np.sum(h_polish_init)
        if s > 0:
            h_polish_init /= s
        
        h_polished, s_polished = _optimize_epigraph(h_polish_init, max_iter=1000, ftol=1e-10)
        if h_polished is not None and s_polished > best_ratio:
            best_ratio = s_polished
            best_h = h_polished.copy()
        
        # Final coordinate descent
        h_ref, s_ref = _coordinate_descent_refinement(h_polished, steps=50)
        if s_ref > best_ratio:
            best_ratio = s_ref
            best_h = h_ref

    # 5. More random restarts if time permits
    while _time_left() > 10.0:
        K = np.random.randint(5, 20)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = _optimize_epigraph(h_rand, max_iter=100)
        if h_opt is not None and s_opt > best_ratio:
            best_ratio = s_opt
            best_h = h_opt.copy()

    if best_h is None:
        return list(np.ones(N) / N)
        
    return list(best_h)
