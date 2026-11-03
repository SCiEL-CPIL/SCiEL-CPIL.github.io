import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # 1. Initialization: Hexagonal Grid
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    points = []
    k = int(1.5 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii)
    points = points[indices[:N]]
    x0 = points.flatten()
    
    def soft_min_obj(params, k):
        c = params.reshape(N, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        radii = np.linalg.norm(c, axis=1)
        bound_dists = 1.0 - radii
        
        vals = np.concatenate([dists.flatten() / 2.0, bound_dists])
        
        v_min = np.min(vals)
        exps = np.exp(-k * (vals - v_min))
        soft_min = v_min - np.log(np.sum(exps)) / k
        
        return -soft_min

    def soft_min_grad(params, k):
        c = params.reshape(N, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        radii = np.linalg.norm(c, axis=1)
        bound_dists = 1.0 - radii
        
        vals = np.concatenate([dists.flatten() / 2.0, bound_dists])
        
        v_min = np.min(vals)
        exps = np.exp(-k * (vals - v_min))
        S = np.sum(exps)
        probs = exps / S
        
        p_pairs = probs[:N*N].reshape(N, N)
        p_bounds = probs[N*N:]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            num = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            den = 2.0 * dists
            V = num / den
            np.fill_diagonal(V, 0.0)
            
        grad_c_pair = np.sum(p_pairs * V, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vecs = c / radii[:, np.newaxis]
            grad_c_bound = p_bounds[:, np.newaxis] * (-unit_vecs)
        
        grad_c = grad_c_pair + grad_c_bound
        return -grad_c.flatten()

    best_x = x0
    best_score = -np.inf
    bounds = [(-1.0, 1.0)] * (2 * N)

    # Multi-start with different k
    for k in [50, 100, 200, 500, 1000]:
        if time.time() > deadline:
            break
            
        for restart in range(3):
            if time.time() > deadline:
                break
                
            if restart == 0:
                x_start = x0
            else:
                # Add noise to escape local minima
                noise = np.random.normal(0, 0.02, x0.shape)
                x_start = x0 + noise
                # Project back to valid region roughly? No, let optimizer handle it.
                
            try:
                res = minimize(
                    soft_min_obj, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    jac=lambda x: soft_min_grad(x, k),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                c = res.x.reshape(N, 2)
                d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
                np.fill_diagonal(d, np.inf)
                min_dist = np.min(d) / 2
                min_bound = np.min(1.0 - np.linalg.norm(c, axis=1))
                score = min(min_dist, min_bound)
                
                if score > best_score:
                    best_score = score
                    best_x = res.x
            except Exception:
                pass

    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                soft_min_obj, 
                best_x, 
                method='L-BFGS-B', 
                bounds=bounds, 
                jac=lambda x: soft_min_grad(x, 2000),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c = res.x.reshape(N, 2)
            d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            min_dist = np.min(d) / 2
            min_bound = np.min(1.0 - np.linalg.norm(c, axis=1))
            score = min(min_dist, min_bound)
            
            if score > best_score:
                best_x = res.x
                best_score = score
        except Exception:
            pass

    return best_x.reshape(N, 2)
