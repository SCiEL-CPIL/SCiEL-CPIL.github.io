"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140.0  # keep a safety margin
    
    # Hexagonal lattice cut to the unit disk: valid but improvable.
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
        if time.time() > deadline:
            return points
    
    # Now refine using force-directed relaxation
    # Objective: maximize min distance (or equivalently minimize overlap)
    # Use a smooth surrogate: sum of exp(-d^2 / sigma^2) for all pairs + boundary penalty
    
    def _compute_score(pts):
        # Compute the actual score: min(min_pair_dist/2, min(1-|c_i|))
        n = len(pts)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        # Distance to boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary)
    
    # Force-directed relaxation with time limit
    pts = points.astype(np.float64)
    current_score = _compute_score(pts)
    
    # Use gradient-based optimization with scipy
    from scipy.optimize import minimize
    
    def objective(x):
        if time.time() > deadline - 5.0:
            return 0.0
        p = x.reshape(-1, 2)
        # Smooth approximation of the min: use log-sum-exp for pair distances
        # and for boundary distances
        n = len(p)
        # Pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, 1e10)
        
        # Use soft-min via log-sum-exp
        # min(d) ≈ -log(sum(exp(-d^2/t))) * t / 2  (for small t)
        t = 0.005
        # For pairs
        exp_neg = np.exp(-dists_sq / t)
        exp_neg[np.eye(n, dtype=bool)] = 0  # remove diagonal
        soft_min_pair = -t * np.log(np.sum(exp_neg)) / 2.0
        
        # For boundary
        radii_sq = np.sum(p * p, axis=1)
        # Distance to boundary is 1 - |p|, so we want to maximize min(1-|p|)
        # Equivalent to minimizing max(|p|)
        # Use soft-max for radii
        exp_r = np.exp(radii_sq / t)
        soft_max_radius = t * np.log(np.sum(exp_r)) / 2.0
        
        # We want to maximize min(soft_min_pair/2, 1-soft_max_radius)
        # Or minimize: max(2*soft_max_radius - 2*soft_min_pair + t2, 2*soft_max_radius - 2 + t2)
        # Actually, let's just minimize a penalty that increases when constraints are violated
        
        # Simple approach: minimize sum of violations
        # Constraint 1: all pair distances >= 2r (where r is our target radius)
        # Constraint 2: all radii <= 1-r
        
        # Let's use a different formulation: minimize (max_radius + sum_of_close_pairs)
        close_pairs_penalty = np.sum(np.exp(-(dists_sq - (2*current_score)**2) / t))
        
        return soft_max_radius + close_pairs_penalty
    
    # Try multiple restarts with different initial perturbations
    best_pts = pts.copy()
    best_score = current_score
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            start_pts = pts.copy()
        else:
            # Perturb the initial points
            start_pts = pts.copy()
            rng = np.random.RandomState(restart)
            start_pts += rng.normal(0, 0.001, size=start_pts.shape)
        
        # Ensure validity
        x0 = start_pts.flatten()
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            if time.time() > deadline - 10.0:
                break
            
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            continue
    
    # Final high-precision polish pass using exact gradient of the min objective
    # The true objective is r = min(min_pair_dist/2, min_boundary)
    # We can do a few iterations of coordinate descent / local search with exact gradients
    
    def _polish(pts, max_iters=50):
        """Polish using exact local optimization: move each point to maximize min distance."""
        pts = pts.copy()
        n = len(pts)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                
                # Find the current bottleneck for point i
                # Compute distances to all other points and to boundary
                dists_to_others = np.linalg.norm(pts - pts[i], axis=1)
                dists_to_others[i] = np.inf
                min_dist_to_others = np.min(dists_to_others)
                argmin_idx = np.argmin(dists_to_others)
                
                r_i = np.linalg.norm(pts[i])
                boundary_dist = 1.0 - r_i
                
                # The local radius limit for point i is min(min_dist_to_others/2, boundary_dist)
                local_r = min(min_dist_to_others / 2.0, boundary_dist)
                
                # Try to move point i slightly to increase local_r
                # Direction: away from the nearest point, and toward center if boundary is limiting
                if min_dist_to_others / 2.0 < boundary_dist:
                    # Move away from nearest point
                    direction = pts[i] - pts[argmin_idx]
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-10:
                        direction = direction / norm_dir
                    else:
                        direction = np.array([1.0, 0.0])
                else:
                    # Move toward center
                    direction = -pts[i]
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-10:
                        direction = direction / norm_dir
                    else:
                        direction = np.array([1.0, 0.0])
                
                # Try several step sizes
                best_step = 0.0
                best_new_r = local_r
                for step in [0.001, 0.0005, 0.0001, 0.002]:
                    new_pos = pts[i] + step * direction
                    # Check if this improves the global score
                    pts_temp = pts.copy()
                    pts_temp[i] = new_pos
                    new_score = _compute_score(pts_temp)
                    if new_score > best_new_r:
                        best_new_r = new_score
                        best_step = step
                
                if best_step > 0:
                    pts[i] += best_step * direction
                    improved = True
            
            if not improved:
                break
        
        return pts
    
    # Apply polish
    polished_pts = _polish(best_pts)
    polished_score = _compute_score(polished_pts)
    if polished_score > best_score:
        best_pts = polished_pts
        best_score = polished_score
    
    return best_pts
# EVOLVE-BLOCK-END
