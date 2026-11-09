"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # New Approach: Square Grid Initialization + L-BFGS-B with Smoothed Min Objective
    
    # 1. Initialization: Square Grid
    # We need 200 points.
    # A 15x15 grid has 225 points.
    # Let's create a 15x15 grid centered at 0.
    n_side = 15
    # Spacing such that the grid fits in [-0.5, 0.5]
    # If we use 15 points, there are 14 intervals.
    # Let's define the grid in a normalized space [-1, 1] and scale later.
    # Actually, let's just define it in [-0.5, 0.5] directly.
    # To maximize initial radius, we want points as far apart as possible.
    # Grid spacing s = 1.0 / (n_side - 1)
    s = 1.0 / (n_side - 1)
    
    grid_x = np.linspace(-0.5, 0.5, n_side)
    grid_y = np.linspace(-0.5, 0.5, n_side)
    xx, yy = np.meshgrid(grid_x, grid_y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # We have 225 points. We need to remove 25.
    # Remove points from the corners/edges to keep the bulk dense?
    # Or just remove random ones?
    # Removing corner points allows the remaining points to expand slightly?
    # Actually, if we remove points, the "effective" area per point increases, allowing larger radius.
    # Let's remove the 25 points with the largest distance from the center?
    # No, that would leave a hole in the middle.
    # Let's remove points from the outer ring.
    # Distance from center:
    dists = np.linalg.norm(points, axis=1)
    # Sort by distance descending
    sort_idx = np.argsort(dists)[::-1]
    # Remove the first 25 (furthest)
    remove_idx = sort_idx[:25]
    keep_idx = np.setdiff1d(np.arange(len(points)), remove_idx)
    points = points[keep_idx]
    
    # 2. Objective Function: Soft Min
    # We want to maximize r = min(min_pair_dist/2, min_wall_dist)
    # Equivalent to minimizing -r.
    # Soft min approximation: -1/k * log(sum(exp(-k * d_i)))
    # This approximates min(d_i) as k -> inf.
    # We use a moderate k to keep it smooth for L-BFGS-B.
    
    def objective(x):
        pts = x.reshape(-1, 2)
        
        # Wall distances
        # dist to wall = 0.5 - |coord|
        # We need min of these.
        wall_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        
        # Pair distances
        # Calculate all pairwise distances
        # To save memory, we can use pdist, but we need the gradient?
        # L-BFGS-B can use numerical gradients if we don't provide one.
        # Providing an analytical gradient for soft-min is complex.
        # Let's rely on numerical gradients, but L-BFGS-B with 400 variables and numerical gradients might be slow.
        # However, 200 points is manageable.
        
        from scipy.spatial.distance import pdist
        pair_dists = pdist(pts)
        
        # Combine all distances (pair and wall)
        # Note: pair_dists are distances between centers.
        # The radius constraint is dist >= 2r, so r <= dist/2.
        # Wall constraint is dist_to_wall >= r, so r <= dist_to_wall.
        # So we want to maximize min( min(pair_dists)/2, min(wall_dists) ).
        
        # Let's define d_i as the "effective radius limit" for each constraint.
        # For pairs: limit = dist/2
        # For walls: limit = dist_to_wall
        
        pair_limits = pair_dists / 2.0
        wall_limits = wall_dists
        
        # Combine into a single array
        all_limits = np.concatenate([pair_limits, wall_limits])
        
        # Soft min
        k = 500.0 # High k for sharper min approximation
        # To avoid underflow/overflow in exp, shift by min
        m = np.min(all_limits)
        if m < 1e-10:
            return 0.0 # Avoid log(0) or similar issues, return large penalty (minimizing -r, so -0 is small? No, we want to minimize -r. If r=0, obj=0. If r>0, obj<0. So we want to go to negative infinity. Wait.
            # Objective is -r.
            # If r is small, -r is close to 0.
            # If r is large, -r is very negative.
            # We want to minimize the objective.
            # So we want the most negative value.
            # Soft min of limits approximates min(limits).
            # So we return -soft_min.
            
        # Soft min calculation
        # log(sum(exp(-k * (x_i - m)))) - k*m ... wait.
        # min(x) approx = -1/k * log(sum(exp(-k*x)))
        # To be numerically stable:
        # sum(exp(-k*x)) = exp(-k*m) * sum(exp(-k*(x-m)))
        # log(sum(exp(-k*x))) = -k*m + log(sum(exp(-k*(x-m))))
        # soft_min = -1/k * (-k*m + log(sum(exp(-k*(x-m))))) = m - 1/k * log(sum(exp(-k*(x-m))))
        
        exp_terms = np.exp(-k * (all_limits - m))
        s = np.sum(exp_terms)
        soft_min = m - (1.0/k) * np.log(s)
        
        return -soft_min

    # 3. Optimization
    x0 = points.flatten()
    
    # Use L-BFGS-B
    # Bounds: [-0.5, 0.5] for all coordinates
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
    
    optimal_pts = res.x.reshape(-1, 2)
    
    # Clip to ensure validity
    optimal_pts[:, 0] = np.clip(optimal_pts[:, 0], -0.5, 0.5)
    optimal_pts[:, 1] = np.clip(optimal_pts[:, 1], -0.5, 0.5)
    
    return optimal_pts
# EVOLVE-BLOCK-END
