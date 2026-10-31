import numpy as np
import time

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave margin
    
    def compute_score(pts):
        """Compute (max_d / min_d)^2 for a set of points"""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-10:
            return 1e10
        return (max_d / min_d)**2
    
    # Generate multiple candidate configurations
    configs = []
    
    # Config 1: Triangular lattice (16 closest to origin)
    R = 6
    lattice_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            lattice_pts.append((x, y))
    lattice_pts = np.array(lattice_pts)
    dists_from_origin = np.sum(lattice_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    config1 = lattice_pts[sorted_indices[:N]]
    configs.append(config1)
    
    # Config 2: Square grid
    grid_size = 4
    grid_pts = []
    for i in range(grid_size):
        for j in range(grid_size):
            grid_pts.append((i - 1.5, j - 1.5))
    grid_pts = np.array(grid_pts)
    configs.append(grid_pts)
    
    # Config 3: Random perturbation of triangular lattice
    np.random.seed(42)
    config3 = config1.copy()
    config3 += np.random.normal(0, 0.1, config3.shape)
    configs.append(config3)
    
    # Config 4: Another random configuration
    np.random.seed(123)
    config4 = config1.copy()
    config4 += np.random.normal(0, 0.3, config4.shape)
    configs.append(config4)
    
    # Optimize each configuration
    best_overall_score = 1e10
    best_overall_params = None
    best_overall_pts = None
    
    for cfg_idx, initial_pts in enumerate(configs):
        if time.time() > deadline - 10:
            break
        
        # Find closest pair to fix scale
        diff = initial_pts[:, np.newaxis, :] - initial_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = initial_pts[i0]
        p1 = initial_pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            continue
        
        # Align p0->(0,0), p1->(1,0)
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (initial_pts - p0) @ R_mat.T / norm_v
        
        # Reorder so i0 is index 0, i1 is index 1
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        init_params = transformed[2:].flatten()
        
        def objective(params, _deadline=deadline):
            if time.time() > _deadline:
                return 1e10
            pts_opt = np.zeros((N, 2))
            pts_opt[0] = [0.0, 0.0]
            pts_opt[1] = [1.0, 0.0]
            pts_opt[2:] = params.reshape(-1, 2)
            return compute_score(pts_opt)
        
        bnds = [(-10, 10)] * (2 * (N - 2))
        
        best_params = init_params
        best_score = objective(init_params)
        
        # Try multiple optimization methods
        for method, opts in [
            ('L-BFGS-B', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}),
            ('Nelder-Mead', {'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}),
        ]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(objective, best_params, method=method, bounds=bnds if method == 'L-BFGS-B' else None,
                               options=opts)
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_params = best_params.copy()
            final_pts = np.zeros((N, 2))
            final_pts[0] = [0.0, 0.0]
            final_pts[1] = [1.0, 0.0]
            final_pts[2:] = best_params.reshape(-1, 2)
            best_overall_pts = final_pts
    
    # If we have a good result, return it
    if best_overall_pts is not None and best_overall_score < 1e9:
        return best_overall_pts
    
    # Fallback: return the best initial config
    if len(configs) > 0:
        return configs[0]
    else:
        return np.random.randn(N, 2)

# EVOLVE-BLOCK-END
