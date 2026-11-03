"""
Optimized packing of 50 equal spheres in a unit cube using smooth surrogate optimization.
Combines grid/random initialization with L-BFGS-B on a log-sum-exp soft-min objective.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0  # seconds, safety margin from 150s limit
TIME_MARGIN = 5.0  # return this many seconds before deadline


def _compute_radius(centers):
    """Compute the exact maximum common radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Boundary constraint: min distance from center to cube face
    # Cube is [-0.5, 0.5]^3
    dist_to_face = np.min(0.5 - np.abs(centers), axis=1)
    r_boundary = np.min(dist_to_face)
    
    # Pairwise constraint: min distance between any two centers / 2
    # Use vectorized calculation for speed
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_boundary, r_pair)


def _smooth_objective(x, beta=50.0):
    """
    Smooth approximation of the negative minimum radius.
    We minimize this, so we want to maximize the radius.
    Uses log-sum-exp for smoothness: -1/beta * log(sum(exp(-beta * val))) approximates -min(val).
    """
    centers = x.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10) # Avoid division by zero or log(0) issues later
    dists = np.sqrt(dists_sq)
    
    # Extract upper triangle to avoid double counting and self-distance
    triu_idx = np.triu_indices(N, k=1)
    pair_dists = dists[triu_idx] / 2.0  # Radius contribution from pair
    
    # Wall values: each coordinate of each center contributes to wall distance
    # Actually, the constraint is min_k (0.5 - |c_ik|) >= r
    # So the values to consider are the individual wall distances for each coordinate?
    # No, the radius is limited by the closest face.
    # dist_to_face_i = min(0.5 - |x_i|, 0.5 - |y_i|, 0.5 - |z_i|)
    # We want min over all i of dist_to_face_i.
    # This is equivalent to min over all i,k of (0.5 - |c_ik|).
    wall_vals = (0.5 - np.abs(centers)).flatten()
    
    # Combine all constraint values
    all_vals = np.concatenate([pair_dists, wall_vals])
    
    # Smooth minimum
    # To avoid overflow in exp, subtract max
    max_val = np.max(all_vals)
    # If max_val is very large, exp(-beta*(val - max)) is safe.
    # If max_val is negative? No, distances are non-negative.
    # If all vals are 0, sum(exp(0)) = N, log(N)/beta.
    
    # Handle potential numerical issues if values are very close to 0 or negative due to floating point
    # But we have bounds, so values should be >= 0.
    
    smooth_min = max_val - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (all_vals - max_val))))
    
    return -smooth_min


def _init_grid(n):
    """Generate an initial grid-based configuration with jitter."""
    # 4x4x4 = 64 points, pick 50
    # Spacing is 1/4 = 0.25. Radius approx 0.125.
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points.append([
                    -0.5 + (i + 0.5) * 0.25,
                    -0.5 + (j + 0.5) * 0.25,
                    -0.5 + (k + 0.5) * 0.25
                ])
    points = np.array(points)
    
    # Randomly select 50 to break symmetry
    idx = np.random.choice(len(points), n, replace=False)
    points = points[idx]
    
    # Add small random jitter to break symmetry and help escape local minima
    points += np.random.randn(n, 3) * 0.005
    points = np.clip(points, -0.49, 0.49)
    return points

def _init_random(n):
    """Generate a random initial configuration."""
    return np.random.uniform(-0.45, 0.45, (n, 3))


def _optimize(points, max_time, start_time):
    """
    Optimize the configuration using L-BFGS-B on the smooth objective.
    Returns best points found.
    """
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = x0.copy()
    best_f = _smooth_objective(x0, beta=50.0)
    
    # Phase 1: Coarse optimization with lower beta
    try:
        res = minimize(
            _smooth_objective,
            x0,
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > max_time:
        return best_x.reshape(N, 3)
        
    # Phase 2: Finer optimization with higher beta
    def _obj_fine(x):
        return _smooth_objective(x, beta=200.0)
        
    try:
        res2 = minimize(
            _obj_fine,
            best_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res2.fun < best_f:
            best_f = res2.fun
            best_x = res2.x
    except Exception:
        pass
        
    return best_x.reshape(N, 3)


def construct():
    start_time = time.time()
    
    # Try multiple initial configurations and local optimizations
    best_centers = None
    best_radius = 0.0
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - TIME_MARGIN - elapsed()
    
    # Define initializations
    np.random.seed(42)
    inits = []
    
    # 1. Grid based
    inits.append(_init_grid(N))
    
    # 2. Random
    inits.append(_init_random(N))
    
    # 3. Layered structure (from original)
    layers = []
    # 5 layers of 10? Or 4 layers?
    # Let's try 5 layers of 10 spheres each.
    for layer_idx in range(5):
        z_pos = (layer_idx + 0.5) / 5.0 - 0.5
        for i in range(10):
            # Arrange in a circle or grid in xy
            # Simple grid 2x5?
            row = i // 5
            col = i % 5
            x = (col + 0.5) / 5.0 - 0.5
            y = (row + 0.5) / 2.0 - 0.5
            layers.append([x, y, z_pos])
    inits.append(np.array(layers[:N]))
    
    # 4. Another random with different seed
    np.random.seed(123)
    inits.append(_init_random(N))
    
    for init_idx, init in enumerate(inits):
        if remaining() < 15.0:
            break
        
        # Allocate time for this initialization
        # We have ~115s total. 4 inits. ~25s each?
        # But L-BFGS-B is fast. Let's give it 20s per init.
        max_opt_time = min(20.0, remaining())
        
        opt_centers = _optimize(init, max_opt_time, start_time)
        
        # Ensure within bounds (L-BFGS-B should do this, but just in case)
        opt_centers = np.clip(opt_centers, -0.5, 0.5)
        
        r = _compute_radius(opt_centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()
            
        # If we have a very good result, try to refine it further with more time
        if best_radius > 0.135:
             # Refinement loop
             for _ in range(2):
                 if remaining() < 5.0:
                     break
                 refined = _optimize(best_centers, min(10.0, remaining()), start_time)
                 refined = np.clip(refined, -0.5, 0.5)
                 r_ref = _compute_radius(refined)
                 if r_ref > best_radius:
                     best_radius = r_ref
                     best_centers = refined

    if best_centers is None:
        # Fallback to grid
        best_centers = _init_grid(N)
    
    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_compute_radius(centers)}")
    print(f"Shape: {centers.shape}")
