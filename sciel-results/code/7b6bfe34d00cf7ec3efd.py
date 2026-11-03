import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 <= 1e-12:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 1e-12:
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
        # Clip h to 0 to handle numerical noise during optimization
        h_clip = np.clip(h, 0, None)
        c = np.convolve(h_clip, h_clip, mode="full")
        return M - c

    def constraints_autoconv_grad(x):
        h = x[:N]
        # Gradient of (M - sum(h_i h_{k-i})) w.r.t h_i
        # Analytical derivative of c_k = sum_i h_i h_{k-i} w.r.t h_i is 2*h_{k-i}
        grad = np.zeros((2 * N - 1, N + 1))
        for k in range(2 * N - 1):
            derivs = np.zeros(N)
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    derivs[i] = 2.0 * h[j]
            grad[k, :N] = -derivs  # We want M - c_k >= 0
            grad[k, N] = 1.0       # grad of M w.r.t M is 1
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
                
            ratio_res = _ratio(h_res)
            return h_res, ratio_res
    except Exception:
        pass
    return None, 0.0

def _find_sidon_support(target_size, N, attempts=100):
    """
    Attempts to find a Sidon set of size target_size in [0, N).
    Returns the set if found, None otherwise.
    """
    for _ in range(attempts):
        candidates = list(range(N))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for cand in candidates:
            if len(S) == target_size:
                break
            
            conflict = False
            new_sums = set()
            
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
                
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums or sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == target_size:
            return S
    return None

def _gaussian_basis(center, width, N):
    """
    Generate a discrete Gaussian-like basis function.
    center: index in [0, N)
    width: standard deviation in indices
    """
    x = np.arange(N)
    g = np.exp(-0.5 * ((x - center) / width) ** 2)
    return g

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
    # Smaller Sidon sets might allow for "purer" structures
    for target_k in [5, 6, 7, 8, 10, 12]:
        if _time_left() < 15.0:
            break
        
        support = _find_sidon_support(target_k, N, attempts=50)
        if support is not None:
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0 / len(support)
            
            # Optimize from this start
            h_opt, s_opt = _optimize_epigraph(h_sidon, max_iter=300)
            if h_opt is not None and s_opt > best_ratio:
                best_ratio = s_opt
                best_h = h_opt.copy()

    # 3. Try Gaussian basis combinations (Parametric Search)
    # This transforms the discrete optimization into a search over centers/widths
    # We try to find a combination of 1-3 Gaussians that minimizes autoconv peak.
    
    def eval_gaussian_combo(params):
        """
        params: list of (center, width)
        """
        h = np.zeros(N)
        for center, width in params:
            h += _gaussian_basis(center, width, N)
        
        s = np.sum(h)
        if s <= 0:
            return None, 0.0
        h = h / s
        
        # Refine with SLSQP starting from this Gaussian shape
        h_opt, s_opt = _optimize_epigraph(h, max_iter=200)
        if h_opt is not None:
            return h_opt, s_opt
        return h, _ratio(h)

    # Single Gaussian
    for _ in range(30):
        if _time_left() < 10.0:
            break
        center = np.random.uniform(0, N)
        width = np.random.uniform(1.0, 5.0)
        h_opt, s_opt = eval_gaussian_combo([(center, width)])
        if s_opt > best_ratio:
            best_ratio = s_opt
            best_h = h_opt

    # Two Gaussians
    for _ in range(20):
        if _time_left() < 10.0:
            break
        c1 = np.random.uniform(0, N/2)
        c2 = np.random.uniform(N/2, N)
        w1 = np.random.uniform(1.0, 4.0)
        w2 = np.random.uniform(1.0, 4.0)
        h_opt, s_opt = eval_gaussian_combo([(c1, w1), (c2, w2)])
        if s_opt > best_ratio:
            best_ratio = s_opt
            best_h = h_opt

    # 4. Try random sparse starts with varying support sizes
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
            best_h = h_opt

    # 5. Polish the best solution found
    # Use tighter tolerances and more iterations on the best candidate
    if best_h is not None and _time_left() > 20.0:
        # Zero out very small values to potentially simplify the problem
        h_polish_init = best_h.copy()
        h_polish_init[h_polish_init < 1e-6] = 0.0
        s = np.sum(h_polish_init)
        if s > 0:
            h_polish_init /= s
        
        h_polished, s_polished = _optimize_epigraph(h_polish_init, max_iter=1000, ftol=1e-10)
        if h_polished is not None and s_polished > best_ratio:
            best_ratio = s_polished
            best_h = h_polished.copy()

    # 6. If we have time, try a few more random restarts with higher K
    while _time_left() > 10.0:
        K = np.random.randint(5, 20)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = _optimize_epigraph(h_rand, max_iter=150)
        if h_opt is not None and s_opt > best_ratio:
            best_ratio = s_opt
            best_h = h_opt

    if best_h is None:
        best_h = np.ones(N) / N
    else:
        # Final cleanup
        best_h = np.clip(best_h, 0, None)
        s = np.sum(best_h)
        if s > 0:
            best_h = best_h / s
        else:
            best_h = np.zeros(N)
            best_h[0] = 1.0

    return list(best_h)
