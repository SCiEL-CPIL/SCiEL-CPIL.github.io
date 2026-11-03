import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50


def construct():
    """
    Construct centers for 50 equal spheres in the unit cube [-0.5, 0.5]^3.
    Uses a BCC lattice initialization followed by repulsive energy minimization
    and final max-min polish.
    """
    np.random.seed(42)
    
    # 1. Generate BCC lattice points
    # A 3x3x3 BCC supercell has 54 points.
    points = []
    n_cells = 3
    a = 1.0 / n_cells
    
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                x0 = i * a
                y0 = j * a
                z0 = k * a
                # Corner
                points.append([x0, y0, z0])
                # Body Center
                points.append([x0 + 0.5*a, y0 + 0.5*a, z0 + 0.5*a])
                
    pts = np.array(points)
    # Shift to center in [-0.5, 0.5]
    pts = pts - 0.5
    
    # We have 54 points. We need 50.
    # Select the 50 points that maximize the minimum distance among themselves
    # using a greedy Farthest Point Sampling approach.
    
    selected_indices = [np.random.choice(len(pts))]
    while len(selected_indices) < N:
        selected_pts = pts[selected_indices]
        unselected_indices = [i for i in range(len(pts)) if i not in selected_indices]
        best_idx = -1
        best_min_dist = -1
        for idx in unselected_indices:
            dists = np.linalg.norm(pts[idx] - selected_pts, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        selected_indices.append(best_idx)
        
    init_pts = pts[selected_indices]
    
    # 2. Optimization using Repulsive Energy
    # Minimize sum of 1/d^2 to spread points evenly and away from boundaries.
    # This creates a smooth landscape for L-BFGS-B.
    
    def repulsive_energy(x):
        p = x.reshape(N, 3)
        # Pairwise distances
        d = cdist(p, p)
        np.fill_diagonal(d, np.inf)
        # Avoid division by zero
        d = np.maximum(d, 1e-6)
        pair_energy = np.sum(1.0 / (d**2))
        
        # Boundary distances
        dist_to_bound = 0.5 - np.max(np.abs(p), axis=1)
        dist_to_bound = np.maximum(dist_to_bound, 1e-6)
        bound_energy = np.sum(1.0 / (dist_to_bound**2))
        
        return pair_energy + bound_energy

    x0 = init_pts.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            repulsive_energy,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        opt_pts = res.x.reshape(N, 3)
    except Exception:
        opt_pts = init_pts

    # 3. Final Polish: Maximize the minimum radius
    # The repulsive energy maximizes average separation.
    # Now we optimize the worst-case (minimum) separation using Powell.
    
    def min_radius_obj(x):
        p = x.reshape(N, 3)
        # Penalty for going out of bounds
        if np.any(np.abs(p) > 0.5 + 1e-9):
            return 1e10
        
        d = cdist(p, p)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        
        dist_to_bound = 0.5 - np.max(np.abs(p), axis=1)
        min_bound = np.min(dist_to_bound)
        
        r = 0.5 * min(min_pair, min_bound)
        return -r

    try:
        res2 = minimize(
            min_radius_obj,
            opt_pts.flatten(),
            method='Powell',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        opt_pts = res2.x.reshape(N, 3)
    except Exception:
        pass

    # Ensure final points are strictly inside the cube
    opt_pts = np.clip(opt_pts, -0.499999, 0.499999)
    
    return opt_pts
