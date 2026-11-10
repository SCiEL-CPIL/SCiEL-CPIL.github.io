import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    n = 8

    def neg_log_score(points):
        # points: (8, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        
        # Extract upper triangle
        i, j = np.triu_indices(n, k=1)
        d2 = sq_dists[i, j]
        
        if np.any(d2 < 1e-12):
            return 1e10
            
        max_d2 = np.max(d2)
        if max_d2 < 1e-12:
            return 1e10
            
        # Objective: - (sum(log(d)) - 28*log(D))
        sum_log_d = 0.5 * np.sum(np.log(d2))
        log_D = 0.5 * np.log(max_d2)
        
        return -(sum_log_d - 28.0 * log_D)

    def two_squares_offset(x):
        # x = [r1, r2, theta_rel, x_off, y_off]
        r1, r2, theta_rel, x_off, y_off = x
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        angles2 = angles1 + theta_rel
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        p2[:, 0] += x_off
        p2[:, 1] += y_off
        
        return np.vstack([p1, p2])

    def obj_sym(x):
        pts = two_squares_offset(x)
        return neg_log_score(pts)

    # 1. Optimize 5-param symmetric manifold
    bounds_sym = [
        (0.1, 5.0),   # r1
        (0.1, 5.0),   # r2
        (0.0, np.pi/2), # theta_rel
        (-1.0, 1.0),  # x_off
        (-1.0, 1.0)   # y_off
    ]

    best_sym_val = 1e10
    best_sym_x = None
    
    try:
        res_sym = differential_evolution(obj_sym, bounds_sym, maxiter=100, popsize=15, tol=1e-8, seed=42, polish=True, init='sobol')
        if res_sym.fun < best_sym_val:
            best_sym_val = res_sym.fun
            best_sym_x = res_sym.x
    except Exception:
        pass

    # Initialize best points
    if best_sym_x is not None:
        best_pts = two_squares_offset(best_sym_x)
    else:
        # Fallback to regular octagon
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    best_val = neg_log_score(best_pts)

    # 2. Full 16-dim optimization
    def obj_full(x):
        pts = x.reshape(8, 2)
        return neg_log_score(pts)

    bounds_full = [(-5.0, 5.0)] * (2 * n)
    x0_full = best_pts.flatten()

    # Polish with L-BFGS-B
    try:
        res_full = minimize(obj_full, x0_full, method='L-BFGS-B', bounds=bounds_full, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
        if res_full.fun < best_val:
            best_val = res_full.fun
            best_pts = res_full.x.reshape(8, 2)
    except Exception:
        pass

    # 3. Random restarts in full space
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        x0_rand = rng.uniform(-2, 2, (8, 2)).flatten()
        try:
            res_rand = minimize(obj_full, x0_rand, method='L-BFGS-B', bounds=bounds_full, options={'maxiter': 300, 'ftol': 1e-12})
            if res_rand.fun < best_val:
                best_val = res_rand.fun
                best_pts = res_rand.x.reshape(8, 2)
        except Exception:
            pass

    # 4. Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(obj_full, best_pts.flatten(), method='SLSQP', bounds=bounds_full, options={'maxiter': 200, 'ftol': 1e-15})
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_pts = res_polish.x.reshape(8, 2)
        except Exception:
            pass

    return [[float(p[0]), float(p[1])] for p in best_pts]
