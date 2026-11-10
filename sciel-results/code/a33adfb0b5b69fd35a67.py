# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _union_length_1d(x, n_quad=50):
    """
    Compute the integral of union length of intervals
    [x_i + c*i/N, x_i + 1/N + c*i/N] for i=0..N-1, c in [0,1].
    Uses Gauss-Legendre quadrature with exact union length at each point.
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


def _union_length_1d_grad(x, n_quad=50):
    """Compute union length integral and its gradient using central differences."""
    N = len(x)
    eps = 1e-7
    f0 = _union_length_1d(x, n_quad)
    grad = np.zeros(N)
    for i in range(N):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        grad[i] = (_union_length_1d(xp, n_quad) - _union_length_1d(xm, n_quad)) / (2 * eps)
    return f0, grad


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 620.0  # safety margin from 630s limit
    
    # Keich binary construction as initial guess
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    def objective(x):
        if time.time() > deadline - 10:
            return 1e10
        return _union_length_1d(x)
    
    def gradient(x):
        if time.time() > deadline - 10:
            return np.zeros(N)
        _, g = _union_length_1d_grad(x)
        return g
    
    # Optimize x-coordinates
    best_x = keich.copy()
    best_fx = objective(best_x)
    
    rng = np.random.default_rng(42)
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            x_init = best_x.copy()
        elif restart == 1:
            x_init = keich.copy()
        else:
            scale = 0.05 / (restart - 1)
            x_init = best_x + rng.normal(0, scale, N)
        
        try:
            res = minimize(
                objective,
                x_init,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_fx:
                best_fx = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Optimize y-coordinates (same problem structure)
    best_y = keich.copy()
    best_fy = objective(best_y)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            y_init = best_y.copy()
        elif restart == 1:
            y_init = keich.copy()
        else:
            scale = 0.05 / (restart - 1)
            y_init = best_y + rng.normal(0, scale, N)
        
        try:
            res = minimize(
                objective,
                y_init,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_fy:
                best_fy = res.fun
                best_y = res.x.copy()
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on the combined 2N-dimensional problem
    if time.time() < deadline - 30:
        def combined_objective(params):
            if time.time() > deadline - 5:
                return 1e10
            px = params[:N]
            py = params[N:]
            # Volume is proportional to integral of Lx(c)*Ly(c) dc
            # Since x and y are independent, we can compute the product integral
            c_points, w_points = np.polynomial.legendre.leggauss(30)
            c_points = 0.5 * (c_points + 1)
            w_points = 0.5 * w_points
            
            vol = 0.0
            for c, w in zip(c_points, w_points):
                x_starts = px + c * np.arange(N) / N
                y_starts = py + c * np.arange(N) / N
                
                # Compute union length for x
                x_ends = x_starts + 1.0 / N
                x_order = np.argsort(x_starts)
                xs = x_starts[x_order]
                xe = x_ends[x_order]
                Lx = 0.0
                cs = xs[0]
                ce = xe[0]
                for k in range(1, N):
                    if xs[k] > ce:
                        Lx += ce - cs
                        cs = xs[k]
                        ce = xe[k]
                    else:
                        ce = max(ce, xe[k])
                Lx += ce - cs
                
                # Compute union length for y
                y_ends = y_starts + 1.0 / N
                y_order = np.argsort(y_starts)
                ys = y_starts[y_order]
                ye = y_ends[y_order]
                Ly = 0.0
                cs = ys[0]
                ce = ye[0]
                for k in range(1, N):
                    if ys[k] > ce:
                        Ly += ce - cs
                        cs = ys[k]
                        ce = ye[k]
                    else:
                        ce = max(ce, ye[k])
                Ly += ce - cs
                
                vol += w * Lx * Ly
            
            return vol
        
        init_params = np.concatenate([best_x, best_y])
        try:
            res = minimize(
                combined_objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < 1e9:  # valid result
                best_x = res.x[:N].copy()
                best_y = res.x[N:].copy()
        except Exception:
            pass
    
    # Build the 2D positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
