import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def construct():
    """
    Construct centers for 50 equal spheres in the unit cube [-0.5, 0.5]^3.
    Uses BCC lattice initialization followed by multi-start local optimization.
    """
    np.random.seed(42)

    # 1. Generate a dense BCC-like candidate set
    # We use a grid of 3x3x3 cells, each with 2 points (corner + center)
    # Total 54 points.
    n_cells = 3
    a = 1.0 / n_cells
    candidates = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                # Corner
                candidates.append([i*a - 0.5, j*a - 0.5, k*a - 0.5])
                # Center
                candidates.append([(i+0.5)*a - 0.5, (j+0.5)*a - 0.5, (k+0.5)*a - 0.5])
    candidates = np.array(candidates)
    
    # 2. Select 50 points from candidates to maximize minimum distance
    # Greedy farthest-point sampling
    selected_indices = [0] # Start with first
    while len(selected_indices) < N:
        selected_pts = candidates[selected_indices]
        unselected_indices = [i for i in range(len(candidates)) if i not in selected_indices]
        best_idx = -1
        best_min_dist = -1
        for idx in unselected_indices:
            dists = np.linalg.norm(candidates[idx] - selected_pts, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        selected_indices.append(best_idx)
    
    x0 = candidates[selected_indices]
    x0 = np.clip(x0, -0.49, 0.49)

    # 3. Optimization
    def objective(x):
        pts = x.reshape(N, 3)
        if np.any(np.abs(pts) > 0.5):
            return 1e10
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_bound = 0.5 - np.max(np.abs(pts), axis=1)
        min_bound_dist = np.min(dist_to_bound)
        r = 0.5 * min(min_pair_dist, min_bound_dist)
        return -r

    best_x = x0.flatten()
    best_f = objective(best_x)

    # Multi-start Powell
    n_restarts = 15
    for i in range(n_restarts):
        if i == 0:
            x_init = x0.flatten()
        else:
            # Perturb
            scale = 0.02 * (i + 1)
            x_init = x0 + np.random.normal(0, scale, N*3)
            x_init = np.clip(x_init.reshape(N,3), -0.49, 0.49).flatten()
        
        try:
            res = minimize(
                objective,
                x_init,
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

    result_centers = best_x.reshape(N, 3)
    result_centers = np.clip(result_centers, -0.499999, 0.499999)
    return result_centers
