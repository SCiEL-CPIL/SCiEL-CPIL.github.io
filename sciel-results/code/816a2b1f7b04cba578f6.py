import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def calc_radius(c):
        if c.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 0.0
        d = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        min_b = np.min(0.5 - np.abs(c))
        return min(min_d / 2.0, min_b)

    def generate_fcc(n_points):
        pts = []
        size = 5
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    pts.append((i, j, k))
                    pts.append((i + 0.5, j + 0.5, k))
                    pts.append((i + 0.5, j, k + 0.5))
                    pts.append((i, j + 0.5, k + 0.5))
        pts = np.array(pts)
        pts -= pts.mean(axis=0)
        max_abs = np.max(np.abs(pts))
        scale = 0.45 / max_abs
        pts *= scale
        np.random.seed(42)
        idx = np.random.choice(len(pts), n_points, replace=False)
        return pts[idx]

    def generate_random(n_points):
        np.random.seed(42)
        return np.random.uniform(-0.45, 0.45, size=(n_points, 3))

    def objective(v, W):
        c = v[:3*N].reshape(N, 3)
        r = v[3*N]
        
        d = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        np.fill_diagonal(d, np.inf)
        
        overlaps = np.maximum(0, 2*r - d)
        penalty_overlap = np.sum(overlaps**2)
        
        b = 0.5 - np.abs(c)
        violations = np.maximum(0, r - b)
        penalty_boundary = np.sum(violations**2)
        
        return -r + W * (penalty_overlap + penalty_boundary)
        
    def gradient(v, W):
        c = v[:3*N].reshape(N, 3)
        r = v[3*N]
        
        grad = np.zeros_like(v)
        grad[3*N] = -1.0
        
        d = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        np.fill_diagonal(d, np.inf)
        
        overlaps = np.maximum(0, 2*r - d)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        
        factor = 2.0 * W * overlaps / d
        np.fill_diagonal(factor, 0.0)
        
        grad_c = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        b = 0.5 - np.abs(c)
        violations = np.maximum(0, r - b)
        signs = np.sign(c)
        grad_b = 2.0 * W * violations * signs
        
        grad[:3*N] = grad_c.reshape(-1) + grad_b.reshape(-1)
        
        grad_r_overlap = 4.0 * W * np.sum(overlaps)
        grad_r_boundary = 2.0 * W * np.sum(violations)
        
        grad[3*N] += grad_r_overlap + grad_r_boundary
        
        return grad

    best_v = None
    best_score = 0.0
    
    inits = [
        generate_fcc(N),
        generate_random(N),
        generate_random(N) # different seed needed?
    ]
    # Set different seeds for random
    np.random.seed(1)
    inits[1] = np.random.uniform(-0.45, 0.45, size=(N, 3))
    np.random.seed(2)
    inits[2] = np.random.uniform(-0.45, 0.45, size=(N, 3))

    W = 1000.0
    bounds = [(-0.5, 0.5)] * (3*N) + [(0.0, 0.5)]
    
    for i, init_c in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        v0 = np.concatenate([init_c.flatten(), [calc_radius(init_c)]])
        
        # Run optimization
        # Use partial to pass W
        from functools import partial
        obj = partial(objective, W=W)
        jac = partial(gradient, W=W)
        
        try:
            res = minimize(
                obj, 
                v0, 
                jac=jac, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            v_new = res.x
            c_new = v_new[:3*N].reshape(N, 3)
            
            if np.all(np.isfinite(c_new)) and np.all(c_new >= -0.5) and np.all(c_new <= 0.5):
                score_new = calc_radius(c_new)
                if score_new > best_score:
                    best_score = score_new
                    best_v = v_new
        except Exception:
            pass

    if best_v is None:
        return generate_fcc(N)
        
    return best_v[:3*N].reshape(N, 3)
