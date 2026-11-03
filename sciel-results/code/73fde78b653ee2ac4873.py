"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def construct():
    """
    Constructs the centers of 300 equal circles inside a unit circle.
    Uses an iterative radius-increasing strategy with L-BFGS-B optimization.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer

    # 1. Initialization
    # Random uniform points in the unit circle
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    best_points = points.copy()
    best_r = 0.0

    # Helper to calculate the current valid radius for a set of points
    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        norms = np.linalg.norm(pts, axis=1)
        min_bound_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_bound_dist)

    # Helper to calculate energy (sum of squared violations) for a target radius
    def energy(pts, target_r):
        # Pairwise overlaps
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount: max(0, 2*target_r - dist)
        overlaps = np.maximum(0, 2.0 * target_r - dists)
        # Sum of squares
        e_pair = np.sum(overlaps ** 2)
        
        # Boundary violations
        norms = np.linalg.norm(pts, axis=1)
        bound_viol = np.maximum(0, norms - (1.0 - target_r))
        e_bound = np.sum(bound_viol ** 2)
        
        return e_pair + e_bound

    # Helper to run L-BFGS-B optimization
    def optimize(points_init, target_r):
        # Flatten points for scipy
        x0 = points_init.flatten()
        
        def obj(x):
            p = x.reshape(-1, 2)
            return energy(p, target_r)
            
        # Gradient is expensive to compute analytically, let's use numerical or just default
        # L-BFGS-B uses numerical gradient by default if not provided, which is slow for N=300 (600 vars).
        # Better to provide gradient?
        # Let's try default first. If too slow, we'll need analytic gradient.
        # Actually, for 600 variables, numerical gradient takes 600 function evals per step.
        # 100 steps = 60000 evals. Each eval is O(N^2) = 90000 ops. Total 5.4e9 ops.
        # This might be too slow in Python loops inside energy.
        # We need to vectorize energy well. It is vectorized.
        # But 60000 calls to energy might take 60000 * (time for one energy).
        # Time for one energy: N^2 distance matrix is 300x300. np.linalg.norm is fast.
        # Let's estimate. 300x300 matrix creation and norm is ~1ms?
        # 60000 * 1ms = 60s. This is borderline.
        
        # To speed up, we can use a smaller number of iterations or a different method.
        # Or we can use 'Powell' which is derivative-free? No, Powell is also slow for high dim.
        # Let's try to provide the gradient analytically.
        
        def grad(x):
            p = x.reshape(-1, 2)
            # Pairwise gradient
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            dists_safe = np.where(dists > 1e-10, dists, 1e-10)
            
            overlaps = np.maximum(0, 2.0 * target_r - dists)
            # d(overlap^2)/d(dist) = 2 * overlap * d(overlap)/d(dist)
            # d(overlap)/d(dist) = -1 if dist < 2r else 0
            # So grad_dist = -2 * overlaps where dist < 2r
            # grad_p_i = sum_j (grad_dist_ij * (p_i - p_j) / dists_ij)
            
            # Vectorized:
            # unit_diff = diff / dists_safe[:, :, np.newaxis]
            # grad_dist = -2.0 * overlaps
            # grad_p = sum_j (grad_dist_ij * unit_diff_ij)
            
            unit_diff = diff / dists_safe[:, :, np.newaxis]
            grad_dist = -2.0 * overlaps
            # Mask diagonal to avoid self-interaction (overlaps is 0 on diag anyway)
            np.fill_diagonal(grad_dist, 0.0)
            
            g_pair = np.sum(grad_dist[:, :, np.newaxis] * unit_diff, axis=1)
            
            # Boundary gradient
            norms = np.linalg.norm(p, axis=1)
            bound_viol = np.maximum(0, norms - (1.0 - target_r))
            # d(bound_viol^2)/d(p) = 2 * bound_viol * d(bound_viol)/d(p)
            # d(bound_viol)/d(p) = p / norms if norms > 1-r
            unit_inward = p / np.maximum(norms[:, np.newaxis], 1e-10)
            g_bound = 2.0 * bound_viol[:, np.newaxis] * unit_inward
            
            return (g_pair + g_bound).flatten()

        res = opt.minimize(obj, x0, method='L-BFGS-B', jac=grad, options={'maxiter': 200, 'ftol': 1e-10})
        return res.x.reshape(-1, 2), res.fun

    # Main loop: Increase target_r
    # Start with a small r
    # Estimate initial r: area based
    # r_est = sqrt(pi / (N * pi)) = 1/sqrt(N) approx? No, area per circle is pi/N.
    # pi * r^2 = pi/N => r = 1/sqrt(N) = 1/sqrt(300) = 0.0577
    # Best known is 0.0531. So start around 0.04.
    
    target_r = 0.03
    step_size = 0.002
    current_points = points.copy()

    while time.time() - start_time < time_limit:
        # Try to optimize for target_r
        optimized_points, energy_val = optimize(current_points, target_r)
        
        # Check if we achieved a valid packing for this target_r
        # If energy is very small, it's valid.
        if energy_val < 1e-8:
            # Calculate actual radius
            actual_r = calc_radius(optimized_points)
            if actual_r > best_r:
                best_r = actual_r
                best_points = optimized_points.copy()
                current_points = optimized_points.copy()
                # Increase target_r
                target_r += step_size
                # Reduce step size slightly as we get closer to limit?
                step_size *= 0.99
            else:
                # If actual_r is not greater than best_r, but energy is low,
                # it means we are at a local minimum for this target_r but didn't improve global best.
                # This shouldn't happen if we start from best_points.
                # Just increase target_r slightly to force expansion.
                target_r += step_size * 0.5
        else:
            # Optimization failed to find a valid configuration for target_r
            # This means target_r is too large or we are stuck in a local min.
            # Perturb and retry with same target_r or decrease target_r.
            # Let's perturb and retry.
            current_points = best_points + np.random.normal(0, 0.01, size=best_points.shape)
            # Pull back if outside
            norms = np.linalg.norm(current_points, axis=1)
            mask = norms > (1.0 - best_r)
            if np.any(mask):
                current_points[mask] = best_points[mask]
            
            # If we keep failing, decrease target_r to escape
            # We can track consecutive failures.
            # For simplicity, just retry. If it fails again, it will loop.
            # To avoid infinite loop, we can decrease target_r after some failures.
            # Let's add a counter.
            # Since we don't have state outside the loop, we'll just perturb.
            # If we are stuck, the energy won't go to 0.
            # Let's try a smaller step or restart.
            # If this happens too often, we might want to reduce target_r.
            # Heuristic: if energy is high, reduce target_r.
            target_r -= step_size * 0.1

    return best_points

# EVOLVE-BLOCK-END
