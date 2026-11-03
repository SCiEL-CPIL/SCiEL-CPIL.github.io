import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    timeout = 110

    def energy(x, r):
        x = x.reshape(N, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        overlap_viol = np.maximum(0, 2*r - dist)
        e_overlap = np.sum(overlap_viol**2)
        
        dist_origin = np.sqrt(np.sum(x**2, axis=1))
        bound_viol = np.maximum(0, dist_origin - (1 - r))
        e_bound = np.sum(bound_viol**2)
        
        return e_overlap + e_bound

    def sa_optimize(x_init, r, max_iter=2000, t_start=1.0, t_end=0.01):
        x = x_init.copy()
        e_curr = energy(x, r)
        best_x = x.copy()
        best_e = e_curr
        
        for i in range(max_iter):
            if time.time() - start_time > timeout:
                break
            t = t_start * (t_end / t_start) ** (i / max_iter)
            
            # Random move
            idx = np.random.randint(N)
            delta = np.random.normal(0, 0.05, 2)
            x_new = x.copy()
            x_new[idx] += delta
            
            e_new = energy(x_new, r)
            
            if e_new < e_curr or np.random.rand() < np.exp(-(e_new - e_curr) / t):
                x = x_new
                e_curr = e_new
                if e_curr < best_e:
                    best_e = e_curr
                    best_x = x.copy()
                    
        return best_x, best_e

    def l_bfgs_polish(x, r):
        def objective(x):
            return energy(x, r)
            
        def gradient(x):
            x = x.reshape(N, 2)
            grad = np.zeros_like(x)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
            np.fill_diagonal(dist, 1.0)
            
            viol = np.maximum(0, 2*r - dist)
            np.fill_diagonal(viol, 0)
            
            term = viol / dist
            np.fill_diagonal(term, 0)
            
            sum_term = np.sum(term, axis=1)
            sum_term_xj = np.sum(term[:, :, np.newaxis] * x[np.newaxis, :, :], axis=1)
            
            grad_overlap = -2 * (x * sum_term[:, np.newaxis] - sum_term_xj)
            
            dist_origin = np.sqrt(np.sum(x**2, axis=1) + 1e-12)
            bound_viol = np.maximum(0, dist_origin - (1 - r))
            grad_bound = 2 * bound_viol[:, np.newaxis] * (x / dist_origin[:, np.newaxis])
            
            return (grad_overlap + grad_bound).flatten()

        res = minimize(
            objective, 
            x, 
            jac=gradient, 
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x.reshape(N, 2), objective(res.x)

    # Initialization
    np.random.seed(42)
    # Start with a grid-like distribution to give SA a head start
    x_init = np.array([
        (i * 0.2, j * 0.2) for i in range(-3, 4) for j in range(-3, 4)
    ])
    radii = np.hypot(x_init[:, 0], x_init[:, 1])
    x_init = x_init[radii < 0.8]
    if len(x_init) >= N:
        x_init = x_init[:N]
    else:
        x_init = np.random.uniform(-0.8, 0.8, (N, 2))

    best_r = 0.0
    best_x = x_init.copy()
    
    r = 0.08
    step = 0.002
    
    while r < 0.14 and time.time() - start_time < timeout:
        # Try to find valid x for this r
        x_candidate, e_candidate = sa_optimize(x_init, r, max_iter=1000)
        
        if e_candidate < 1e-5:
            # Valid!
            # Polish with L-BFGS
            x_polished, e_polished = l_bfgs_polish(x_candidate, r)
            if e_polished < 1e-5:
                best_r = r
                best_x = x_polished
                r += step
            else:
                # L-BFGS failed, accept SA result if it's good enough
                best_r = r
                best_x = x_candidate
                r += step
        else:
            # Invalid, try smaller step or break
            # If we just started, maybe r is too high
            # If we increased, maybe we hit a wall
            # Try to recover by reducing r slightly and re-optimizing?
            # For now, break or try a different init
            break
            
    # Final polish
    if best_r > 0:
        for _ in range(20):
            if time.time() - start_time > timeout:
                break
            r_test = best_r + 0.0005
            x_test, e_test = l_bfgs_polish(best_x, r_test)
            if e_test < 1e-6:
                best_r = r_test
                best_x = x_test
            else:
                break

    if best_x is None or best_r == 0:
        # Fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    return best_x
