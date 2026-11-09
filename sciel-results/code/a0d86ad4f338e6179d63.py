# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area_fast(x, n=64, M=20000):
    """
    Fast quadrature-based area computation (matches the evaluator exactly).
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    return float(area)

def _soft_min_area(x, n=64, M=20000, beta=50.0):
    """
    Smooth surrogate for the area using soft-min.
    min(a,b) ≈ -log(exp(-beta*a) + exp(-beta*b)) / beta
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Soft-min of d and w: -log(exp(-beta*d) + exp(-beta*w))/beta
    # Use log-sum-exp trick for stability
    beta_d = beta * d
    beta_w = beta * w[np.newaxis, :]
    # min(a,b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # = -1/beta * (max + log(exp(-(beta*a - max)) + exp(-(beta*b - max))))
    # where max = max(beta*a, beta*b)
    mx = np.maximum(beta_d, beta_w)
    sm = -mx / beta - np.log(np.exp(-beta_d - mx) + np.exp(-beta_w - mx)) / beta
    
    union_lengths = w[np.newaxis, :] + np.sum(sm, axis=0)
    area = np.mean(union_lengths)
    return float(area)

def _soft_min_area_grad(x, n=64, M=20000, beta=50.0):
    """
    Compute soft-min area and its gradient.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted, order = np.sort(L, axis=0), np.argsort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    beta_d = beta * d
    beta_w = beta * w[np.newaxis, :]
    mx = np.maximum(beta_d, beta_w)
    
    # Soft-min value
    exp_neg_d = np.exp(-beta_d - mx)
    exp_neg_w = np.exp(-beta_w - mx)
    sm = -mx / beta - np.log(exp_neg_d + exp_neg_w) / beta
    
    union_lengths = w[np.newaxis, :] + np.sum(sm, axis=0)
    area = float(np.mean(union_lengths))
    
    # Gradient: d/dx_i of sm
    # sm = -1/beta * log(exp(-beta*d) + exp(-beta*w))
    # d/dx_i sm = -1/beta * [(-beta * d/dx_i d) * exp(-beta*d) + 0] / (exp(-beta*d) + exp(-beta*w))
    #           = d/dx_i d * exp(-beta*d) / (exp(-beta*d) + exp(-beta*w))
    #           = d/dx_i d * sigmoid(beta*(w-d))
    # where d/dx_i d = a_{order[j+1,m]} - a_{order[j,m]} for the gap j at height m
    # (since d_j = L_sorted[j+1] - L_sorted[j], and L_i = x_i + a_i*y)
    
    # sigmoid(beta*(w - d))
    arg = beta * (w[np.newaxis, :] - d)
    sig = 1.0 / (1.0 + np.exp(-np.clip(arg, -500, 500)))
    
    # sig has shape (n-1, M)
    # For gap j at height m: d/dx_i d_j = a_{order[j+1,m]} - a_{order[j,m]}
    # grad[i] = sum_m sig[j,m] * (a_{order[j+1,m]} - a_{order[j,m]}) * I(i == order[j+1,m])
    #          + sum_m sig[j,m] * (a_{order[j+1,m]} - a_{order[j,m]}) * I(i == order[j,m]) * (-1)
    
    da = a[order[1:, :]] - a[order[:-1, :]]  # shape (n-1, M)
    contrib = sig * da  # shape (n-1, M)
    
    grad = np.zeros(n, dtype=np.float64)
    # For each gap j and height m, add contrib[j,m] to grad[order[j+1,m]] and subtract from grad[order[j,m]]
    j_idx = np.arange(n - 1)[:, np.newaxis]
    m_idx = np.arange(M)[np.newaxis, :]
    upper = order[j_idx + 1, m_idx]  # shape (n-1, M)
    lower = order[j_idx, m_idx]  # shape (n-1, M)
    
    np.add.at(grad, upper.ravel(), contrib.ravel())
    np.add.at(grad, lower.ravel(), -contrib.ravel())
    
    grad /= M
    return area, grad

def _subgrad_area_grad(x, n=64, M=20000):
    """
    Compute area and subgradient using the fixed-ordering trick.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted, order = np.sort(L, axis=0), np.argsort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = float(np.mean(union_lengths))
    
    sat = (d < w[np.newaxis, :])  # shape (n-1, M)
    
    grad = np.zeros(n, dtype=np.float64)
    sat_indices = np.where(sat)
    if len(sat_indices[0]) > 0:
        j_idx = sat_indices[0]
        m_idx = sat_indices[1]
        upper = order[j_idx + 1, m_idx]
        lower = order[j_idx, m_idx]
        np.add.at(grad, upper, 1.0)
        np.add.at(grad, lower, -1.0)
    
    grad /= M
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 600.0
    
    a = (np.arange(n) + 1) / n
    
    # Initial guesses
    inits = [
        np.zeros(n),
        -a * 0.5,
        -a * (1.0/3.0),
        -a * (2.0/3.0),
        a * 0.5,
        -a * 0.25,
        -a * 0.75,
    ]
    
    best_x = inits[0].copy()
    best_area = _compute_area_fast(inits[0], n, M)
    
    for init in inits[1:]:
        area_init = _compute_area_fast(init, n, M)
        if area_init < best_area:
            best_area = area_init
            best_x = init.copy()
    
    # Phase 1: Smooth gradient-based optimization using soft-min surrogate
    if time.time() < deadline - 120:
        try:
            from scipy.optimize import minimize
            
            def smooth_obj(x):
                return _soft_min_area(x, n, M, beta=100.0)
            
            def smooth_grad(x):
                _, g = _soft_min_area_grad(x, n, M, beta=100.0)
                return g
            
            # Try multiple beta values, starting coarse
            for beta_val in [50.0, 100.0, 200.0, 500.0]:
                if time.time() > deadline - 120:
                    break
                
                def obj_b(x):
                    return _soft_min_area(x, n, M, beta=beta_val)
                
                def grad_b(x):
                    _, g = _soft_min_area_grad(x, n, M, beta=beta_val)
                    return g
                
                t_max = min(deadline - 120, start_time + 80)
                res = minimize(obj_b, best_x, jac=grad_b, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10,
                                     'maxls': 50, 'maxfun': 1000})
                
                if res.fun < 1e10:  # valid result
                    area_exact = _compute_area_fast(res.x, n, M)
                    if area_exact < best_area:
                        best_area = area_exact
                        best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Subgradient descent on exact objective for local refinement
    if time.time() < deadline - 60:
        x = best_x.copy()
        area, g = _subgrad_area_grad(x, n, M)
        v = np.zeros(n, dtype=np.float64)
        mu = 0.9
        step = 0.05
        best_local = area
        best_local_x = x.copy()
        
        t_max = min(deadline - 60, start_time + 90)
        while time.time() < t_max:
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-14:
                break
            step_polyak = max(np.dot(x, g), 1e-12) / (gnorm * gnorm + 1e-12)
            step = min(step_polyak, 0.3)
            v = mu * v - step * g
            xn = x + v
            an, gn = _subgrad_area_grad(xn, n, M)
            if an < area:
                x, area, g = xn, an, gn
                if an < best_local:
                    best_local = an
                    best_local_x = x.copy()
            else:
                step *= 0.7
                v *= 0.5
                if step < 1e-12:
                    break
        
        if best_local < best_area:
            best_area = best_local
            best_x = best_local_x.copy()
    
    # Phase 3: Coordinate descent with adaptive step for final polish
    x = best_x.copy()
    current_area = best_area
    
    for iteration in range(50):
        if time.time() > deadline - 30:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 30:
                break
            
            best_local = current_area
            best_delta = 0.0
            for delta in [-0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02]:
                x_trial = x.copy()
                x_trial[i] += delta
                area_trial = _compute_area_fast(x_trial, n, M)
                if area_trial < best_local:
                    best_local = area_trial
                    best_delta = delta
            
            if best_delta != 0.0:
                x[i] += best_delta
                current_area = best_local
                improved = True
        
        if not improved:
            # Try smaller steps
            improved_small = False
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                for delta in [-0.001, -0.0005, 0.0005, 0.001]:
                    x_trial = x.copy()
                    x_trial[i] += delta
                    area_trial = _compute_area_fast(x_trial, n, M)
                    if area_trial < current_area - 1e-10:
                        x[i] += delta
                        current_area = area_trial
                        improved_small = True
                        break
            if not improved_small:
                break
        
        if current_area < best_area:
            best_area = current_area
            best_x = x.copy()
    
    # Phase 4: Basin hopping with smooth gradient polishing
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for hop in range(10):
            if time.time() > deadline - 30:
                break
            perturbation_scale = 0.2 / (1.0 + hop * 0.15)
            x_pert = best_x + rng.normal(0, perturbation_scale, n)
            area_pert = _compute_area_fast(x_pert, n, M)
            
            # Short L-BFGS-B from perturbed point
            if time.time() < deadline - 30:
                try:
                    def obj_b(x):
                        return _soft_min_area(x, n, M, beta=200.0)
                    def grad_b(x):
                        _, g = _soft_min_area_grad(x, n, M, beta=200.0)
                        return g
                    res = minimize(obj_b, x_pert, jac=grad_b, method='L-BFGS-B',
                                 options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10,
                                         'maxls': 30, 'maxfun': 300})
                    if res.fun < 1e10:
                        area_exact = _compute_area_fast(res.x, n, M)
                        if area_exact < best_area:
                            best_area = area_exact
                            best_x = res.x.copy()
                except Exception:
                    pass
    
    # Phase 5: Final Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            def objective(x):
                return _compute_area_fast(x, n, M)
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_area:
                best_x = res.x.copy()
                best_area = res.fun
        except Exception:
            pass
    
    # Validation
    if not np.all(np.isfinite(best_x)):
        return np.zeros(n, dtype=np.float64)
    if best_area <= 0:
        return np.zeros(n, dtype=np.float64)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
