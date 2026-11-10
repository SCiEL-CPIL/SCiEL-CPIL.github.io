import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        full_path = np.vstack([P_start, path, P_end])
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        return np.array(dense_points)

    def check_inside(placements):
        valid_mask = np.ones(n_points, dtype=bool)
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = X_flat - x
            dy = Y_flat - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = in_rect1 | in_rect2
            
            valid_mask &= inside
            if not np.any(valid_mask):
                return 0
        return np.count_nonzero(valid_mask)

    def score_path(path):
        dense = densify_path(path)
        count = check_inside(dense)
        return count * 0.0001

    def path_to_params(path):
        return path.flatten()

    def params_to_path(params):
        n_pts = len(params) // 3
        path = params.reshape(n_pts, 3)
        return path

    def neg_score(params):
        path = params_to_path(params)
        return -score_path(path)

    # Initial Candidates
    seed_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_path = seed_path
    best_score = score_path(best_path)
    
    rng = np.random.default_rng(42)
    
    def generate_random_path(n_pts):
        angles = np.sort(rng.uniform(5, 85, n_pts))
        xs = rng.uniform(-1.2, 0.5, n_pts)
        ys = rng.uniform(-0.5, 1.5, n_pts)
        path = np.column_stack([xs, ys, angles])
        return path

    # Try different numbers of points
    for n_pts in [5, 6, 7, 8, 9]:
        if time.time() > deadline - 120: break
        
        # Generate several random starts for this n_pts
        for _ in range(3):
            if time.time() > deadline - 100: break
            
            path = generate_random_path(n_pts)
            params = path_to_params(path)
            
            # Define bounds
            bounds = []
            for i in range(n_pts):
                bounds.append((-1.5, 0.8)) # x
                bounds.append((-0.5, 1.5)) # y
                bounds.append((5, 85))     # angle
            
            # Optimize
            try:
                res = minimize(neg_score, params, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < -best_score:
                    new_path = params_to_path(res.x)
                    # Ensure angles are sorted (L-BFGS-B might break order slightly if not constrained strictly)
                    idx = np.argsort(new_path[:, 2])
                    new_path = new_path[idx]
                    
                    new_score = score_path(new_path)
                    if new_score > best_score:
                        best_score = new_score
                        best_path = new_path
            except Exception:
                pass

    # Final polish with Nelder-Mead or more L-BFGS-B iterations
    if time.time() < deadline - 10:
        params = path_to_params(best_path)
        n_pts = len(best_path)
        bounds = []
        for i in range(n_pts):
            bounds.append((-1.5, 0.8))
            bounds.append((-0.5, 1.5))
            bounds.append((5, 85))
        
        try:
            res = minimize(neg_score, params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < -best_score:
                new_path = params_to_path(res.x)
                idx = np.argsort(new_path[:, 2])
                new_path = new_path[idx]
                new_score = score_path(new_path)
                if new_score > best_score:
                    best_score = new_score
                    best_path = new_path
        except Exception:
            pass

    return best_path
