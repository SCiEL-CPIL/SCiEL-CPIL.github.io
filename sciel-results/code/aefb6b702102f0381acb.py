import numpy as np
from scipy.optimize import minimize

N = 300

def construct():
    def calc_min_radius(pts):
        dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dists_boundary)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    def obj_simple(params):
        pts = params.reshape(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        pen_b = np.sum(np.maximum(0, norms - 1.0)**2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        dists[dists == 0] = 1.0
        
        # Sum of inverse squares to maximize min distance
        # Using 0.01 offset to prevent singularity
        val = np.sum(1.0 / (dists + 0.01)**2)
        return val + 100.0 * pen_b

    best_params = None
    best_score = -1.0

    # Multiple random restarts
    for seed in range(10):
        np.random.seed(seed)
        theta = np.random.uniform(0, 2 * np.pi, N)
        r = np.sqrt(np.random.uniform(0, 1, N))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        init_params = np.column_stack((x, y)).flatten()
        
        # Use BFGS
        try:
            res = minimize(obj_simple, init_params, method='BFGS', options={'maxiter': 100, 'disp': False})
            pts = res.x.reshape(N, 2)
            
            # Ensure validity
            norms = np.linalg.norm(pts, axis=1)
            outside = norms > 1.0
            if np.any(outside):
                pts[outside] /= norms[outside, np.newaxis]
                
            score = calc_min_radius(pts)
            if score > best_score:
                best_score = score
                best_params = pts
        except Exception:
            continue

    if best_params is None:
        # Fallback
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.sqrt(np.linspace(0.1, 0.9, N))
        best_params = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
    return best_params
