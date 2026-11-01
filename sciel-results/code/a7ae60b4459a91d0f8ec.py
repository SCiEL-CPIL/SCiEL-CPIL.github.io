"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Base lattice: 5x5x2 grid
    # This exactly matches N=50.
    nx, ny, nz = 5, 5, 2
    
    # Generate the base grid points
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    base_points = []
    for z in zs:
        for y in ys:
            for x in xs:
                base_points.append([x, y, z])
    base_points = np.array(base_points)
    
    # Normalize base points to have unit norm for better scaling of parameters
    norms = np.linalg.norm(base_points, axis=1, keepdims=True)
    # Avoid division by zero (though center point is not in this grid, but good practice)
    norms[norms == 0] = 1.0
    unit_base = base_points / norms
    
    # Parameters to optimize:
    # 1. Rotation matrix (6 params: quaternion or Euler angles? Quaternion is safer for global opt)
    #    Let's use a quaternion (4 params) to define rotation.
    # 2. Translation (3 params)
    # 3. Scale (1 param) - to adjust overall size
    # 4. Breathing (50 params) - radial displacement for each point
    #    Total: 4 + 3 + 1 + 50 = 58 parameters.
    
    def get_centers(params):
        q = params[0:4]
        t = params[4:7]
        scale = params[7]
        breath = params[8:8+N]
        
        # Normalize quaternion
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            q = np.array([1, 0, 0, 0])
        else:
            q = q / q_norm
        
        # Convert quaternion to rotation matrix
        # w, x, y, z
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Apply rotation, scale, and breathing
        # Breathing is applied radially to the base points
        # p_i' = R * (scale * p_i + breath_i * unit_p_i) + t
        # Wait, the prompt says "per-point radial breathing".
        # Let's interpret this as: p_i' = R * (scale * (p_i + breath_i * unit_p_i)) + t
        # Or simply: p_i' = R * (scale * p_i) + breath_i * R * unit_p_i + t
        # Let's stick to a simple radial offset in the rotated frame:
        # p_i_rot = R * (scale * p_i)
        # p_i_final = p_i_rot + breath_i * R * unit_p_i + t
        
        rotated_base = (scale * base_points) @ R.T  # (N, 3) @ (3, 3) -> (N, 3)
        
        # Apply breathing
        # unit_base is the direction from origin to base point
        # We want to move along this direction.
        # The direction in the rotated frame is R * unit_base
        rotated_units = unit_base @ R.T
        
        centers = rotated_base + breath[:, np.newaxis] * rotated_units + t
        
        return centers

    def objective(params):
        centers = get_centers(params)
        return -_compute_radius(centers)

    # Bounds for parameters
    # Quaternion components: [-1, 1]
    # Translation: [-1, 1] (since cube is [-0.5, 0.5], but rotation/scale might push it out, so wider bounds)
    # Scale: [0.5, 2.0]
    # Breathing: [-0.1, 0.1]
    
    bounds = [(-1, 1)] * 4 + [(-1, 1)] * 3 + [(0.5, 2.0)] + [(-0.1, 0.1)] * N
    
    # Initial best guess: Identity rotation, zero translation, scale 1, zero breathing
    best_params = np.zeros(58)
    best_params[0] = 1.0  # w=1 for identity
    best_params[7] = 1.0  # scale=1
    
    best_radius = _compute_radius(get_centers(best_params))
    
    # Run Differential Evolution
    # maxiter is set high, but we rely on the timeout guard inside the worker if we were doing it manually.
    # DE in scipy doesn't have a direct timeout, but we can check time in the objective function?
    # No, DE calls objective many times. If we return a bad value after timeout, it might not stop.
    # Better strategy: Run DE with a limited number of iterations or use a wrapper that checks time.
    # However, DE with 58 dims and 50 points might be slow.
    # Let's try a smaller population and maxiter, and hope it finishes.
    # If it takes too long, we return the best found so far.
    # Scipy's DE doesn't support early stopping via time easily without hacking the objective.
    # Let's hack the objective to track time and return a very large penalty if we exceed 80% of time budget,
    # and then check if we should stop. But DE doesn't stop early.
    
    # Alternative: Use a simpler optimizer like Nelder-Mead on the reduced parameters?
    # Or use COBYLA?
    # Given the structural change directive, let's try to make the DE run as fast as possible.
    # Pop size: 15 * 58 = 870 evaluations per generation.
    # Each eval is ~50x50 distance calc = 2500 ops. Very fast.
    # 100 generations = 87000 evals. Should be < 1s.
    # So DE should be fine.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            seed=42, 
            tol=1e-6,
            polish=True  # Use L-BFGS-B to polish the result
        )
        
        if res.fun < -best_radius:  # Since objective is -radius
            best_radius = -res.fun
            best_params = res.x
        else:
            # Ensure we keep the best from the initial guess if DE didn't improve
            pass
            
    except Exception as e:
        # If DE fails, fall back to the initial guess or a simple local search
        pass

    # Final Polish with Nelder-Mead on the best parameters found
    # This helps with the non-smoothness of the min-distance function
    centers_best = get_centers(best_params)
    
    def nm_objective(params):
        return -_compute_radius(get_centers(params))
    
    try:
        from scipy.optimize import minimize
        res_nm = minimize(
            nm_objective, 
            best_params, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if res_nm.fun < -best_radius:
            best_radius = -res_nm.fun
            best_params = res_nm.x
    except Exception:
        pass

    final_centers = get_centers(best_params)
    
    # Ensure centers are within the cube?
    # The optimizer might push them out if the penalty isn't strong enough in the objective.
    # _compute_radius handles out-of-bounds by giving a negative radius (or 0?).
    # Let's check _compute_radius implementation.
    # If a center is outside, wall_dist is negative.
    # So the objective will naturally penalize it.
    # But to be safe, clip the final result?
    # No, clipping might break the "rigid" assumption or create overlaps.
    # The objective function should keep them inside.
    
    return final_centers

def _compute_radius(centers):
    """Compute the maximum radius for the given centers."""
    N = len(centers)
    
    # Check for non-finite values
    if not np.all(np.isfinite(centers)):
        return -1.0

    # Check if centers are within the cube [-0.5, 0.5]^3
    # If any center is outside, the radius is effectively 0 or negative.
    # The formula min_i min_k (1/2 - |c_ik|) will be negative.
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    
    # Min pair distance
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # Distance to walls: min(0.5 - x, 0.5 + x) = 0.5 - |x|
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    # Radius is half the min distance
    r = min(min_pair_dist, min_wall_dist) / 2.0
    
    return r

# EVOLVE-BLOCK-END
