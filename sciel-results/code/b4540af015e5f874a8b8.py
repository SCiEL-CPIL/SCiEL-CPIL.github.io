# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _union_length_1d(x, n_quad=30):
    """
    Compute the exact integral of union length of intervals
    [x_i + c*i/N, x_i + 1/N + c*i/N] for i=0..N-1, c in [0,1].
    Uses Gauss-Legendre quadrature.
    """
    N = len(x)
    c_points, w_points = np.polynomial.legendre.leggauss(n_quad)
    c_points = 0.5 * (c_points + 1)  # map to [0,1]
    w_points = 0.5 * w_points
    
    total = 0.0
    for c, w in zip(c_points, w_points):
        starts = x + c * np.arange(N) / N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        union_len = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, N):
            if s[k] > cur_end:
                union_len += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
            else:
                cur_end = max(cur_end, e[k])
        union_len += cur_end - cur_start
        total += w * union_len
    
    return total

def _union_length_grad(x, n_quad=30):
    """
    Compute gradient of the union length integral using central finite differences.
    More efficient: compute value and gradient together.
    """
    N = len(x)
    eps = 1e-6
    f0 = _union_length_1d(x, n_quad)
    grad = np.zeros(N)
    for i in range(N):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (_union_length_1d(xp, n_quad) - _union_length_1d(xm, n_quad)) / (2 * eps)
    return f0, grad

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s limit
    
    # Keich construction as initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    x0 = keich.copy()
    best_x = x0.copy()
    best_f = _union_length_1d(x0)
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Simulated annealing with gradient steps
    # Model positions as particles; use gradient of union length as repulsive force
    # with annealing to escape local minima
    
    T0 = 0.05  # initial temperature
    T_final = 0.001  # final temperature
    n_anneal_steps = 2000
    step_size_base = 0.02
    
    x_curr = x0.copy()
    f_curr = best_f
    
    for step in range(n_anneal_steps):
        if time.time() > deadline - 60:
            break
        
        # Annealing schedule: exponential decay
        alpha = step / max(n_anneal_steps - 1, 1)
        T = T0 * (T_final / T0) ** alpha
        step_size = step_size_base * (1.0 - 0.5 * alpha)
        
        # Compute gradient
        f_curr, grad = _union_length_grad(x_curr, n_quad=20)
        
        # Gradient step with noise (annealing perturbation)
        noise = rng.normal(0, T * step_size, N)
        x_new = x_curr - step_size * grad + noise
        
        # Evaluate new position
        f_new = _union_length_1d(x_new)
        
        # Accept/reject based on Metropolis criterion
        if f_new < f_curr or rng.random() < np.exp(-(f_new - f_curr) / max(T, 1e-10)):
            x_curr = x_new
            f_curr = f_new
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
    
    # Phase 2: Polished L-BFGS-B optimization from best found so far
    if time.time() < deadline - 30:
        try:
            res = minimize(
                lambda x: _union_length_1d(x, n_quad=30),
                best_x.copy(),
                jac=lambda x: _union_length_grad(x, n_quad=30)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Additional restarts if time permits
    if time.time() < deadline - 20:
        n_extra_restarts = 3
        for r in range(n_extra_restarts):
            if time.time() > deadline - 15:
                break
            x_init = best_x + rng.normal(0, 0.05, N)
            try:
                res = minimize(
                    lambda x: _union_length_1d(x, n_quad=30),
                    x_init,
                    jac=lambda x: _union_length_grad(x, n_quad=30)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _union_length_1d(x, n_quad=40),
                best_x.copy(),
                jac=lambda x: _union_length_grad(x, n_quad=40)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Build the 2D positions from the optimized 1D positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
