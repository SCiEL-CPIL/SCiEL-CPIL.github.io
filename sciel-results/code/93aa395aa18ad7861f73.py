"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Base lattice: 5x5x2 grid
    # This exactly matches N=50.
    nx, ny, nz = 5, 5, 2
    
    # Base coordinates in [-0.5, 0.5]
    # For optimal packing, we want to maximize spacing.
    # A standard grid in [-0.5, 0.5] with n points has spacing 1/(n-1) if we include boundaries,
    # but spheres can't touch boundaries if r>0.
    # It's better to think of the lattice in [0, 1] or centered.
    # Let's define base points in [-0.5, 0.5] with uniform spacing.
    # Spacing dx = 1/(nx-1) if we span the whole box? 
    # Actually, for a grid of n points, if we want them centered, 
    # positions are -0.5 + (i+0.5)/n ? No, that's for bins.
    # Let's use linspace(-0.5, 0.5, n).
    # For n=5: -0.5, -0.25, 0, 0.25, 0.5.
    # Distance between neighbors is 0.25.
    # Distance to wall is 0.5.
    # This leaves large gaps at walls.
    # A better base lattice for packing might be one where points are not on the boundary.
    # Let's use a "centered" grid:
    # x_i = -0.5 + (i + 0.5) * (1.0 / nx) ?
    # i=0: -0.5 + 0.1 = -0.4
    # i=4: -0.5 + 4.5*0.2 = -0.5 + 0.9 = 0.4
    # Spacing: 0.2.
    # Wall dist: 0.1.
    # This is much better balanced.
    
    xs = -0.5 + (np.arange(nx) + 0.5) * (1.0 / nx)
    ys = -0.5 + (np.arange(ny) + 0.5) * (1.0 / ny)
    zs = -0.5 + (np.arange(nz) + 0.5) * (1.0 / nz)
    
    base_points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # base_points shape: (50, 3)
    
    # Parameters to optimize:
    # 1. Scaling factors for x, y, z (3 params). 
    #    We scale the base points. 
    #    s_x, s_y, s_z. 
    #    To ensure points stay in box, we need to be careful.
    #    Let's parameterize as: p_i = s * base_i + shift?
    #    Actually, the base points are already centered.
    #    If we scale by s > 1, they move out. If s < 1, they move in.
    #    We want to find the optimal s_x, s_y, s_z.
    #    However, the "optimal" packing might not be a simple scaled grid.
    #    The "breathing" allows individual points to move radially.
    
    # Let's use a more flexible parameterization:
    # Each point i has a displacement vector d_i.
    # But 50*3 = 150 params is a lot for L-BFGS-B if we use a soft surrogate?
    # L-BFGS-B can handle 150 params easily.
    # But the "rigid body" idea suggests fewer params.
    # Let's stick to the prompt's suggestion: "rigid point cloud undergoing continuous rigid-body transformations... combined with per-point radial breathing".
    
    # Rigid body: Rotation R (3 params: euler angles) and Translation t (3 params).
    # But the cube is symmetric. Rotating a grid inside a cube usually breaks wall constraints unless rotation is 90 deg multiples.
    # So rigid rotation is likely not useful for a grid aligned with axes.
    # The prompt says "exploit global symmetries... prevent optimizer from getting trapped in asymmetric local minima".
    # Maybe the "breathing" is the key.
    
    # Let's try:
    # 1. Affine scaling of the grid: 3 params (sx, sy, sz).
    # 2. Per-point radial breathing: 50 params (r_i).
    #    Displacement of point i is r_i * (base_i / ||base_i||).
    #    Wait, base_i might be zero.
    #    Let's use a fixed direction vector for each point, or just allow free displacement but regularized?
    #    No, "radial breathing" implies moving along the vector from center to point.
    #    Points at center (0,0,0) have undefined radial direction.
    #    In a 5x5x2 grid, there is no point exactly at (0,0,0) if we use the centered bin method?
    #    xs: -0.4, -0.2, 0, 0.2, 0.4.
    #    zs: -0.25, 0.25.
    #    So there are points with x=0, y=0, z!=0.
    #    Vector is (0,0,z). Radial direction is (0,0,1) or (0,0,-1).
    
    # Let's define the parameter vector theta:
    # theta[0:3] = [sx, sy, sz]
    # theta[3:53] = [r_0, ..., r_49]
    
    # Initial guess:
    # sx=sy=sz=1.0
    # r_i = 0.0
    
    theta0 = np.zeros(53)
    theta0[0:3] = 1.0
    
    def _get_centers(theta):
        sx, sy, sz = theta[0], theta[1], theta[2]
        rs = theta[3:]
        
        # Scale base points
        # base_points are in [-0.5, 0.5]
        # Scaling: p_scaled = base * [sx, sy, sz]
        p_scaled = base_points * np.array([sx, sy, sz])
        
        # Radial breathing
        # Direction for each point
        # norm = ||base_points||
        # If norm is 0, direction is (0,0,0).
        norms = np.linalg.norm(base_points, axis=1)
        # Avoid division by zero
        norms_safe = np.where(norms < 1e-9, 1.0, norms)
        dirs = base_points / norms_safe[:, np.newaxis]
        
        # Apply breathing
        # p_final = p_scaled + r * dir
        p_final = p_scaled + rs[:, np.newaxis] * dirs
        
        return p_final

    def _compute_radius(centers):
        """Compute the maximum radius for the given centers."""
        min_dist = np.inf
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        r = min(min_pair_dist, min_wall_dist) / 2.0
        return r

    def _objective(theta):
        # We want to maximize the radius.
        # L-BFGS-B minimizes. So we minimize -radius.
        # However, radius is non-smooth (min function).
        # We use a soft-min surrogate for smoothness.
        # f = -softmin(distances)
        
        centers = _get_centers(theta)
        
        # Check validity: points must be in [-0.5, 0.5]
        # If not, return a large penalty.
        if np.any(centers < -0.5) or np.any(centers > 0.5):
            # Penalty proportional to violation
            viol = np.maximum(0, np.abs(centers) - 0.5)
            return 100.0 + np.sum(viol)
            
        # Calculate all constraints (distances to pairs and walls)
        # We need the minimum of these distances.
        
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Flatten unique pairs
        # To be efficient, we can just take all i<j
        # But for soft-min, we can just use all upper triangle
        iu = np.triu_indices(N, k=1)
        pair_dists_sq = dists_sq[iu]
        
        # Wall distances
        wall_dists_sq = (0.5 - np.abs(centers)) ** 2
        # Flatten wall dists (3 per point)
        wall_dists_sq_flat = wall_dists_sq.flatten()
        
        # Combine all squared distances
        all_dists_sq = np.concatenate([pair_dists_sq, wall_dists_sq_flat])
        
        # Soft-min of squared distances
        # softmin(x, beta) = -1/beta * log(sum(exp(-beta * x)))
        # As beta -> inf, softmin -> min.
        # We want to maximize min(dist) = sqrt(min(dist_sq)).
        # So we maximize softmin(dist_sq).
        # Minimize -softmin(dist_sq).
        
        beta = 50.0 # Sharpness
        # For numerical stability, subtract max
        # We are doing -log(sum(exp(-beta*x)))
        # Let y = -beta * x
        # sum(exp(y))
        # max_y = max(y)
        # log(sum(exp(y - max_y))) + max_y
        
        y = -beta * all_dists_sq
        max_y = np.max(y)
        # If max_y is very large positive, exp overflows.
        # But y is negative (since dist_sq > 0).
        # So exp(y) is small.
        # Actually, if dist_sq is small, y is close to 0.
        # If dist_sq is large, y is very negative.
        # So max_y is the least negative value (closest to 0).
        
        # To be safe:
        if not np.isfinite(max_y):
             return 100.0
            
        log_sum = np.log(np.sum(np.exp(y - max_y))) + max_y
        softmin_sq = -log_sum / beta
        
        # We want to maximize sqrt(softmin_sq).
        # So minimize -sqrt(softmin_sq).
        
        if softmin_sq < 0:
             softmin_sq = 0.0
             
        return -np.sqrt(softmin_sq)

    # Optimization
    best_theta = theta0.copy()
    best_val = _objective(best_theta)
    
    # Try a few different initial scales?
    # The 5x5x2 grid is quite anisotropic.
    # Maybe we should start with a more isotropic grid if possible?
    # But N=50 is fixed. 5x5x2 is the only "full" rectangular grid.
    # 4x4x4 has 64. We'd have to remove 14.
    # Let's stick to 5x5x2 but allow scaling to fix aspect ratio.
    
    options = {
        'maxiter': 2000,
        'ftol': 1e-8,
        'gtol': 1e-6,
        'maxls': 50
    }
    
    # Run L-BFGS-B
    try:
        res = minimize(_objective, theta0, method='L-BFGS-B', options=options)
        if res.success or res.fun < best_val:
            best_theta = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # Not enough time for polish
        centers = _get_centers(best_theta)
        return centers
    
    # Local Polish: 
    # The soft-min might not be sharp enough to find the exact min.
    # We can do a local search on the actual radius.
    # Since the parameter space is low-dimensional (53), we can do a simple gradient-free local search
    # or use the previous gradient info.
    
    # Let's do a simple coordinate ascent on the actual radius for the last few parameters.
    # Or just use the result from L-BFGS-B.
    # The soft-min with beta=50 is quite sharp.
    
    # One more refinement: 
    # Use the centers from best_theta and run a few iterations of "repulsion" in the parameter space?
    # No, let's just trust L-BFGS-B with the sharp soft-min.
    
    # However, the "radial breathing" might allow points to cross each other or violate order.
    # The soft-min handles overlaps by penalizing small distances.
    
    centers = _get_centers(best_theta)
    
    # Final validation and clipping (just in case)
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers

# EVOLVE-BLOCK-END
