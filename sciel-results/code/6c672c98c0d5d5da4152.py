"""
Fifty equal circles in a unit square - Binary Search with L-BFGS on Violation.
Uses a binary search on the radius 'r' and minimizes the sum of squared constraint violations
to determine feasibility. L-BFGS is used for the inner optimization loop.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill


def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))

    if min_wall < 0:
        return 0.0

    # Use KDTree for efficient nearest neighbor search
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0

    return min(min_pair, min_wall)


def _violation_objective(x, r):
    """
    Calculate the sum of squared violations for a given radius r.
    x is a flat array of length 2N.
    """
    c = x.reshape(-1, 2)
    r2 = 2 * r
    
    # Wall violations
    # We want |x| <= 0.5 - r. Violation is max(0, |x| - (0.5 - r))
    wall_limit = 0.5 - r
    viol_x = np.maximum(0, np.abs(c[:, 0]) - wall_limit)
    viol_y = np.maximum(0, np.abs(c[:, 1]) - wall_limit)
    wall_pen = np.sum(viol_x**2) + np.sum(viol_y**2)
    
    # Pairwise violations
    # We want dist >= 2r. Violation is max(0, 2r - dist)
    # To be fast, we only check neighbors? No, for gradient we need all pairs if we do dense.
    # But N=50 is small. 50*49/2 = 1225 pairs.
    # We can compute all pairwise distances.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask upper triangle
    triu_idx = np.triu_indices(N, k=1)
    pair_dists = dists[triu_idx]
    violations = np.maximum(0, r2 - pair_dists)
    pair_pen = np.sum(violations**2)
    
    return wall_pen + pair_pen


def _violation_gradient(x, r):
    """
    Calculate the gradient of the violation objective.
    """
    c = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    r2 = 2 * r
    wall_limit = 0.5 - r
    
    # Wall gradients
    # d/dx (max(0, |x| - L)^2) = 2 * max(0, |x| - L) * sign(x)
    viol_x = np.maximum(0, np.abs(c[:, 0]) - wall_limit)
    viol_y = np.maximum(0, np.abs(c[:, 1]) - wall_limit)
    
    grad[0::2] += 2 * viol_x * np.sign(c[:, 0])
    grad[1::2] += 2 * viol_y * np.sign(c[:, 1])
    
    # Pairwise gradients
    # We need to sum contributions from all j for each i.
    # This is O(N^2) which is fine for N=50.
    for i in range(N):
        for j in range(i + 1, N):
            d_vec = c[i] - c[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-10:
                # Avoid division by zero, push them apart slightly in gradient?
                # Or just skip, random perturbation in init handles this usually.
                continue
            
            viol = r2 - d
            if viol > 0:
                # d/dc_i (viol^2) = 2 * viol * d/dc_i (r2 - ||c_i - c_j||)
                # d/dc_i (r2 - d) = - (c_i - c_j) / d
                # So term is 2 * viol * (-(c_i - c_j) / d)
                factor = 2 * viol / d
                grad[2*i:2*i+2] -= factor * d_vec
                grad[2*j:2*j+2] += factor * d_vec
                
    return grad


def _try_feasibility(centers_init, r, max_time, max_iter=200):
    """
    Try to find a configuration with radius r using L-BFGS.
    Returns (success, centers, min_violation).
    """
    x0 = centers_init.flatten().copy()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def fun(x):
        return _violation_objective(x, r)
        
    def jac(x):
        return _violation_gradient(x, r)
        
    start_time = time.time()
    
    # Check initial violation
    v0 = fun(x0)
    if v0 < 1e-12:
        return True, centers_init, v0
        
    try:
        result = opt.minimize(
            fun, x0, jac=jac, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-8,
                     'maxfun': max_iter * 2, 'disp': False}
        )
        
        final_v = result.fun
        final_centers = result.x.reshape(-1, 2)
        
        # Ensure centers are within bounds (L-BFGS-B should keep them, but clip just in case)
        final_centers = np.clip(final_centers, -0.5, 0.5)
        
        # Re-evaluate to be sure
        if _compute_radius(final_centers) >= r - 1e-9:
            return True, final_centers, final_v
            
        return False, final_centers, final_v
        
    except Exception:
        return False, centers_init, v0


def _initial_hexagonal():
    """Create initial hexagonal packing layout."""
    # Standard hexagonal grid layout for 50 circles
    # Rows: 8, 7, 8, 7, 8, 7, 5 = 50
    row_counts = [8, 7, 8, 7, 8, 7, 5]
    points = []
    total_rows = len(row_counts)
    # Vertical spacing needs to be tight.
    # For hex packing, vertical dist = sqrt(3)/2 * horizontal dist.
    # Let horizontal spacing be s.
    # Total width approx 8s + (7-1)*s/2? No, staggered.
    # Let's just place them on a grid and let optimization fix it.
    
    # Simple grid first, then perturb
    # 7 rows, approx 8 cols
    rows = 7
    cols = 8
    xs = np.linspace(-0.45, 0.45, cols)
    ys = np.linspace(-0.45, 0.45, rows)
    
    grid_points = []
    for i in range(rows):
        for j in range(cols):
            grid_points.append([xs[j], ys[i]])
            
    # We have 56 points, need 50. Remove corners or just take first 50?
    # Taking first 50 from row-major order might be biased.
    # Let's shuffle and take 50? Or just use a specific layout.
    
    # Better: Use the known good layout structure roughly.
    # Row 0: 8
    # Row 1: 7
    # Row 2: 8
    # Row 3: 7
    # Row 4: 8
    # Row 5: 7
    # Row 6: 5
    
    points = []
    vert_spacing = 0.9 / (total_rows - 1)
    for row_idx, count in enumerate(row_counts):
        y = -0.45 + row_idx * vert_spacing
        horiz_spacing = 0.9 / count if count > 1 else 0
        x_start = -0.45 + (0.9 - horiz_spacing * (count - 1)) / 2
        if row_idx % 2 == 1:
            x_offset = horiz_spacing / 2
        else:
            x_offset = 0
        for i in range(count):
            x = x_start + i * horiz_spacing + x_offset
            points.append([x, y])

    points = np.array(points[:N])
    rng = np.random.default_rng(42)
    points = points + rng.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.49, 0.49)
    return points


def _initial_grid():
    """Create a simple grid initial layout."""
    # 50 points. 7x7 is 49. 8x7 is 56.
    # Let's use 7 rows, 8 cols, drop 6 random ones?
    # Or just 50 random points?
    # Grid is better for local optimization.
    
    rows = 7
    cols = 8
    xs = np.linspace(-0.45, 0.45, cols)
    ys = np.linspace(-0.45, 0.45, rows)
    
    points = []
    for i in range(rows):
        for j in range(cols):
            points.append([xs[j], ys[i]])
            
    # We have 56. Remove the 6 corner ones?
    # Indices to remove: (0,0), (0,7), (6,0), (6,7), and two more?
    # Let's just take the first 50 in a staggered way?
    # Actually, let's just use the hexagonal one as primary init.
    # This function is a backup.
    
    # Select 50 from grid
    indices = np.arange(len(points))
    rng = np.random.default_rng(123)
    selected = rng.choice(indices, N, replace=False)
    return np.array(points)[selected]


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial configuration
    best_centers = _initial_hexagonal()
    best_radius = _compute_radius(best_centers)
    
    # Binary search setup
    low = 0.0700
    high = 0.0720
    
    # First, check if our init is already better than 'low'
    if best_radius > low:
        low = best_radius
        # If it's really high, update high?
        # We know the max is ~0.07137.
        # If best_radius > 0.07137, we are in trouble (unlikely).
        
    # We will perform binary search steps.
    # Each step involves a L-BFGS run.
    # We need to manage time carefully.
    
    current_centers = best_centers.copy()
    current_r = low
    
    # We want to find the highest feasible r.
    # Strategy:
    # 1. Start with r = low.
    # 2. Try to increase r.
    
    # Actually, standard binary search is safer.
    # But L-BFGS is expensive.
    # Let's do a few fixed steps first to get a good baseline.
    
    # Step 1: Polish the initial hexagonal grid to find its optimal radius.
    # We can do this by minimizing the violation for a slightly smaller r than the computed one?
    # No, let's just use the binary search loop.
    
    # To make it faster, we can use a smaller N for gradient? No.
    # We can use fewer L-BFGS iterations.
    
    num_search_steps = 10
    for step in range(num_search_steps):
        if time.time() > deadline - 20:
            break
            
        mid_r = (low + high) / 2
        
        # Time budget for this step
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed - 5 # 5s safety
        steps_remaining = num_search_steps - step
        time_per_step = time_remaining / steps_remaining
        
        # Cap the time per step
        time_per_step = min(time_per_step, 10.0)
        
        # Try to find feasible config for mid_r
        success, centers_trial, viol = _try_feasibility(
            current_centers, mid_r, max_time=time_per_step, max_iter=100
        )
        
        if success:
            low = mid_r
            current_centers = centers_trial
            current_r = mid_r
            # Update best
            r_calc = _compute_radius(centers_trial)
            if r_calc > best_radius:
                best_radius = r_calc
                best_centers = centers_trial
        else:
            high = mid_r
            # If we failed, we might have improved the configuration slightly towards feasibility?
            # L-BFGS minimizes violation. Even if not fully feasible, the config is "better" for that r.
            # But for the next lower r, it might be a good start?
            # Usually, we keep the last feasible config as the start point for the next try.
            # However, if we failed, we know high is infeasible.
            # We should probably restart from the last known good config or a fresh one?
            # Using the config from the failed run as start for next (lower) r is often good because it's "packed".
            current_centers = centers_trial
            
    # After binary search, we have a good candidate in [low, high] where low is feasible.
    # Let's do a final polish on 'low' to squeeze out more precision.
    # We can try to increase 'low' slightly in small increments.
    
    if time.time() < deadline - 10:
        # Try to push the radius a bit more
        epsilon = 0.00001
        attempts = 0
        while time.time() < deadline - 5 and attempts < 50:
            r_try = low + epsilon
            success, centers_trial, viol = _try_feasibility(
                current_centers, r_try, max_time=2.0, max_iter=50
            )
            if success:
                low = r_try
                current_centers = centers_trial
                r_calc = _compute_radius(centers_trial)
                if r_calc > best_radius:
                    best_radius = r_calc
                    best_centers = centers_trial
                epsilon *= 0.9
            else:
                epsilon *= 0.5
                if epsilon < 1e-7:
                    break
            attempts += 1

    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_hexagonal()

    return best_centers

# Test run
if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Best Radius: {r}")
