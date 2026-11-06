"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def construct():
    """
    Returns an array of shape (N, 2) with the centers of N circles.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # The optimal radius is ~0.1258, so diameter ~0.2516.
    # Hex lattice spacing should be slightly larger than diameter.
    # Let's try a few spacings to find a good initial count.
    
    def hex_lattice_init(spacing):
        points = []
        # Range of x and y to cover unit circle
        max_r = 1.0
        k = int(max_r / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= (1.0 - spacing/2.0)**2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        # Sort by distance to center, take closest N
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        return points[idx]
    
    # Find a good initial spacing
    # We want roughly 50 points. Area of unit circle is pi.
    # Area per hex point is spacing^2 * sqrt(3)/2.
    # 50 * spacing^2 * sqrt(3)/2 ~= pi  => spacing^2 ~= 2*pi/(50*sqrt(3)) ~= 0.072 => spacing ~= 0.27
    # Optimal diameter is 0.2516, so spacing 0.27 is reasonable.
    
    best_init = None
    best_init_score = -1
    
    for spacing in [0.26, 0.27, 0.28, 0.29]:
        pts = hex_lattice_init(spacing)
        if pts is not None:
            # Calculate score for this initial config
            r = compute_radius(pts)
            if r > best_init_score:
                best_init_score = r
                best_init = pts.copy()
    
    if best_init is None:
        # Fallback: random points
        best_init = np.random.rand(N, 2) * 2 - 1
        mask = np.linalg.norm(best_init, axis=1) <= 0.9
        while mask.sum() < N:
            new_pts = np.random.rand(100, 2) * 2 - 1
            new_mask = np.linalg.norm(new_pts, axis=1) <= 0.9
            best_init = np.vstack([best_init, new_pts[new_mask]])
            best_init = best_init[:N]
            mask = np.linalg.norm(best_init, axis=1) <= 0.9

    current_best = best_init.copy()
    best_score = best_init_score

    # 2. Optimization using L-BFGS-B
    # Objective: maximize min distance
    # We minimize -min_distance
    
    def objective(x):
        # Reshape to (N, 2)
        pts = x.reshape(-1, 2)
        # Calculate all pairwise distances
        # d_ij = ||p_i - p_j||
        # Efficient calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        radii = np.linalg.norm(pts, axis=1)
        min_boundary_dist = 1.0 - np.max(radii)
        
        # The actual radius is half the min of these two
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return -r

    def objective_soft(x, tau=0.01):
        """
        Soft-min surrogate for smoother gradients.
        Minimizing this is equivalent to maximizing a soft-min of the distances.
        """
        pts = x.reshape(-1, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        radii = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - radii
        
        # Combine all constraints into a single vector of "violations" or "distances"
        # We want to maximize the minimum of these distances.
        # Soft min: -log(sum(exp(-d/tau)))
        
        # Pairwise distances
        # Flatten upper triangle
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        
        # Boundary distances
        all_dists = np.concatenate([pair_dists, boundary_dists])
        
        # Avoid log(0) or issues if dist is negative (shouldn't happen if we constrain)
        # If a point is outside, boundary_dist is negative.
        # exp(-negative/tau) = exp(positive/tau) which is large.
        # This effectively penalizes violations.
        
        # Clip to avoid overflow
        exponents = -all_dists / tau
        exponents = np.clip(exponents, -700, 700)
        
        log_sum = np.log(np.sum(np.exp(exponents)))
        
        # We want to minimize this value?
        # If d is large, exp(-d/tau) is small, sum is small, log is negative.
        # If d is small, exp(-d/tau) is large, sum is large, log is positive.
        # So minimizing log_sum maximizes the minimum distance.
        
        return log_sum

    # Try multiple restarts with perturbations
    restarts = 5
    
    for i in range(restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            x0 = best_init.flatten()
        else:
            # Perturb the best found so far
            # Add small random noise
            noise_scale = 0.05
            x0 = current_best.flatten() + np.random.randn(N*2) * noise_scale
            # Ensure points stay inside roughly
            # Simple projection: if norm > 0.9, scale down
            pts = x0.reshape(-1, 2)
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 0.9
            pts[mask] = pts[mask] * (0.9 / norms[mask])
            x0 = pts.flatten()
        
        # Optimize
        # Use L-BFGS-B
        # We use the soft objective for better gradients
        # Then verify with the hard objective
        
        res = minimize(objective_soft, x0, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        x_opt = res.x.reshape(-1, 2)
        
        # Calculate the true score (hard min)
        score = compute_radius(x_opt)
        
        if score > best_score:
            best_score = score
            current_best = x_opt.copy()
            
    # Final polish: Run one more optimization on the best result found
    # using the hard objective with finite differences, just to be sure
    if time.time() < deadline - 5:
        x0 = current_best.flatten()
        res_final = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        x_final = res_final.x.reshape(-1, 2)
        score_final = compute_radius(x_final)
        
        if score_final > best_score:
            best_score = score_final
            current_best = x_final

    return current_best

def compute_radius(pts):
    """
    Computes the radius of the largest equal circles that fit in the unit circle
    given the centers in pts.
    """
    N = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.linalg.norm(pts, axis=1)
    min_boundary_dist = 1.0 - np.max(radii)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

# EVOLVE-BLOCK-END
