import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import expit

# Constants
N = 50
DELTA = 0.5 / N
_DEADLINE = 135.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio_discrete(h):
    """
    Compute the ratio R = sum(h^2) / (sum(h) * max(conv(h,h))).
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
    Perform coordinate descent to maximize the ratio with finer grid search.
    """
    h = h.copy()
    current_ratio = _ratio_discrete(h)
    
    for _ in range(steps):
        improved = False
        for i in range(N):
            h_i_old = h[i]
            best_h_i = h_i_old
            best_r = current_ratio
            
            # Finer grid: try a range of multipliers
            if h_i_old > 1e-10:
                scales = np.linspace(0.0, 3.0, 16)
            else:
                scales = np.linspace(0.0, 1.0, 16)
            
            for scale in scales:
                h_test = h.copy()
                h_test[i] = h_i_old * scale
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
            
    s = np.sum(h)
    if s > 0:
        h /= s
    
    return h, _ratio_discrete(h)

def _gaussian_basis(num_bumps, N):
    """
    Create a set of Gaussian basis functions on [0, N-1].
    Returns a matrix B of shape (N, num_bumps) where B[i,j] = exp(-((i - c_j)^2)/(2*sigma^2)).
    """
    centers = np.linspace(0, N - 1, num_bumps)
    sigma = N / (2 * num_bumps)
    indices = np.arange(N)
    B = np.zeros((N, num_bumps))
    for j, c in enumerate(centers):
        B[:, j] = np.exp(-((indices - c) ** 2) / (2 * sigma ** 2))
    return B

def _optimize_smooth_basis(num_bumps, h_init=None, max_iter=200):
    """
    Parametrize h = B @ w where B is a Gaussian basis matrix and w >= 0.
    Optimize w to maximize the ratio, subject to sum(h) = 1.
    """
    B = _gaussian_basis(num_bumps, N)
    
    def ratio_from_w(w):
        h = B @ w
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s <= 0:
            return 0.0
        h = h / s
        return _ratio_discrete(h)
    
    def neg_ratio(w):
        return -ratio_from_w(w)
    
    # Initial weights
    if h_init is not None:
        # Project h_init onto the basis
        h_init = np.asarray(h_init, dtype=np.float64)
        s = np.sum(h_init)
        if s > 0:
            h_init = h_init / s
        # Least squares projection with nonnegativity
        try:
            w0, _, _, _ = np.linalg.lstsq(B, h_init, rcond=None)
            w0 = np.clip(w0, 0, None)
        except:
            w0 = np.ones(num_bumps) / num_bumps
    else:
        w0 = np.ones(num_bumps) / num_bumps
    
    # Normalize so that sum(B @ w0) = 1
    h0 = B @ w0
    h0 = np.clip(h0, 0, None)
    s0 = np.sum(h0)
    if s0 > 0:
        w0 = w0 / s0
    
    bounds = [(0.0, None)] * num_bumps
    
    try:
        res = minimize(neg_ratio, w0, method='Nelder-Mead', 
                      options={'maxiter': max_iter * num_bumps, 'xatol': 1e-8, 'fatol': 1e-8})
        
        w_opt = np.clip(res.x, 0, None)
        h_opt = B @ w_opt
        h_opt = np.clip(h_opt, 0, None)
        s = np.sum(h_opt)
        if s > 0:
            h_opt = h_opt / s
        else:
            return None, 0.0
        
        r_opt = _ratio_discrete(h_opt)
        return h_opt, r_opt
    except Exception:
        pass
    return None, 0.0

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

    # 2. Try smooth basis optimization with various numbers of bumps
    for num_bumps in [3, 4, 5, 6, 8]:
        if _time_left() < 15.0:
            break
        
        h_smooth, s_smooth = _optimize_smooth_basis(num_bumps, h_init=best_h, max_iter=100)
        if h_smooth is not None and s_smooth > best_ratio:
            best_ratio = s_smooth
            best_h = h_smooth.copy()
        
        # Polish with epigraph
        if h_smooth is not None:
            h_polished, s_polished = _optimize_epigraph(h_smooth, max_iter=200)
            if h_polished is not None and s_polished > best_ratio:
                best_ratio = s_polished
                best_h = h_polished.copy()
            
            # Coordinate descent refinement
            h_ref, s_ref = _coordinate_descent_refinement(best_h, steps=15)
            if s_ref > best_ratio:
                best_ratio = s_ref
                best_h = h_ref

    # 3. Try Sidon-like structured supports
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
            if h_opt is not None:
                h_ref, s_ref = _coordinate_descent_refinement(h_opt, steps=10)
                if s_ref > best_ratio:
                    best_ratio = s_ref
                    best_h = h_ref

    # 4. Try random sparse starts
    num_restarts = 15
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
            
        if h_opt is not None:
            h_ref, s_ref = _coordinate_descent_refinement(h_opt, steps=10)
            if s_ref > best_ratio:
                best_ratio = s_ref
                best_h = h_ref

    # 5. Polish the best solution
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
        h_ref, s_ref = _coordinate_descent_refinement(best_h, steps=50)
        if s_ref > best_ratio:
            best_ratio = s_ref
            best_h = h_ref

    # 6. More random restarts if time permits
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
