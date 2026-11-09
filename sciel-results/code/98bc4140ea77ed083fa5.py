import numpy as np
import time
from scipy.optimize import minimize

def _union_area(x, n=64, M=20000):
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    return area

def _subgrad_descent_step(x, n=64, M=20000, step_size=0.01, max_iter=200):
    """
    Coordinate-wise sub-gradient descent for the non-smooth min(gap, w) landscape.
    For each coordinate, compute the sub-gradient by finite difference and take a step.
    """
    x = x.copy()
    y = (np.arange(M) + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            # Compute area at current x
            L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
            L_sorted = np.sort(L, axis=0)
            d = L_sorted[1:, :] - L_sorted[:-1, :]
            min_d_w = np.minimum(d, w[np.newaxis, :])
            union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
            area_cur = np.mean(union_lengths)
            
            # Perturb coordinate i
            x_pert = x.copy()
            x_pert[i] += step_size
            L_pert = x_pert[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
            L_sorted_pert = np.sort(L_pert, axis=0)
            d_pert = L_sorted_pert[1:, :] - L_sorted_pert[:-1, :]
            min_d_w_pert = np.minimum(d_pert, w[np.newaxis, :])
            union_lengths_pert = w[np.newaxis, :] + np.sum(min_d_w_pert, axis=0)
            area_pert = np.mean(union_lengths_pert)
            
            # Sub-gradient estimate
            grad_est = (area_pert - area_cur) / step_size
            if grad_est > 0:
                # Move in negative gradient direction
                x[i] -= step_size * 0.5
                improved = True
            elif grad_est < 0:
                x[i] += step_size * 0.5
                improved = True
        
        if not improved:
            break
    
    return x

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 630s limit
    
    # Initial solution: Keich construction
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    best_x = x0.copy()
    best_area = _union_area(x0, n, M)
    
    # Phase 1: Nelder-Mead direct search (handles non-smoothness well)
    if time.time() < deadline - 120:
        try:
            result_nm = minimize(
                lambda x: _union_area(x, n, M),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if result_nm.fun < best_area:
                best_x = result_nm.x.copy()
                best_area = result_nm.fun
        except Exception:
            pass
    
    # Phase 2: Sub-gradient coordinate descent polish
    if time.time() < deadline - 60:
        x_sg = _subgrad_descent_step(best_x, n, M, step_size=0.005, max_iter=100)
        area_sg = _union_area(x_sg, n, M)
        if area_sg < best_area:
            best_x = x_sg.copy()
            best_area = area_sg
    
    # Phase 3: Multiple Nelder-Mead restarts with diverse initializations
    np.random.seed(42)
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Diverse initialization: mix of Keich, random, and sorted uniform
        init_type = restart % 3
        if init_type == 0:
            # Keich + small noise
            x_start = x0 + 0.1 * np.random.randn(n)
        elif init_type == 1:
            # Random
            x_start = np.random.uniform(-0.5, 0.5, n)
        else:
            # Sorted uniform
            x_start = np.sort(np.random.uniform(-0.5, 0.5, n))
        
        try:
            result = minimize(
                lambda x: _union_area(x, n, M),
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            
            if result.fun < best_area:
                best_x = result.x.copy()
                best_area = result.fun
                
                # Polish with sub-gradient descent if time permits
                if time.time() < deadline - 30:
                    x_pol = _subgrad_descent_step(best_x, n, M, step_size=0.003, max_iter=50)
                    area_pol = _union_area(x_pol, n, M)
                    if area_pol < best_area:
                        best_x = x_pol.copy()
                        best_area = area_pol
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_x)):
        return x0
    if best_area <= 0:
        return x0
    
    return best_x.astype(np.float64)
