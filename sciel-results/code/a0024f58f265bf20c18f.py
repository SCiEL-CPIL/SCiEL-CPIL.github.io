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

    # 1. Initialization: Use hexagonal lattice as starting point
    def generate_hexagonal_init(n):
        """Generate initial points using hexagonal packing pattern."""
        points = []
        # Start from center, spiral outward
        # Hexagonal lattice spacing
        spacing = 0.05  # Will be adjusted
        # Generate points in hexagonal pattern
        for ring in range(0, 50):
            if ring == 0:
                points.append([0.0, 0.0])
            else:
                # Number of points in this ring
                n_in_ring = 6 * ring
                for i in range(n_in_ring):
                    angle = 2 * np.pi * i / n_in_ring
                    r = ring * spacing
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    points.append([x, y])
            if len(points) >= n:
                break
        
        points = np.array(points[:n])
        # Scale to fit in unit circle
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0:
            points = points / max_norm * 0.95
        return points

    points = generate_hexagonal_init(N)
    
    best_points = points.copy()
    best_r = 0.0

    # Helper to calculate the current valid radius for a set of points
    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        min_bound_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_bound_dist)

    # Helper to calculate energy (sum of squared violations) for a target radius
    def energy(pts, target_r):
        # Pairwise overlaps
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount: max(0, 2*target_r - dist)
        overlaps = np.maximum(0, 2.0 * target_r - dists)
        # Sum of squares
        e_pair = np.sum(overlaps ** 2)
        
        # Boundary violations
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
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
            
        def grad(x):
            p = x.reshape(-1, 2)
            # Pairwise gradient
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            dists_safe = np.where(dists > 1e-10, dists, 1e-10)
            
            overlaps = np.maximum(0, 2.0 * target_r - dists)
            
            unit_diff = diff / dists_safe[:, :, np.newaxis]
            grad_dist = -2.0 * overlaps
            # Mask diagonal to avoid self-interaction (overlaps is 0 on diag anyway)
            np.fill_diagonal(grad_dist, 0.0)
            
            g_pair = np.sum(grad_dist[:, :, np.newaxis] * unit_diff, axis=1)
            
            # Boundary gradient
            norms = np.sqrt(np.sum(p ** 2, axis=1))
            bound_viol = np.maximum(0, norms - (1.0 - target_r))
            # d(bound_viol^2)/d(p) = 2 * bound_viol * d(bound_viol)/d(p)
            # d(bound_viol)/d(p) = p / norms if norms > 1-r
            unit_inward = p / np.maximum(norms[:, np.newaxis], 1e-10)
            g_bound = 2.0 * bound_viol[:, np.newaxis] * unit_inward
            
            return (g_pair + g_bound).flatten()

        res = opt.minimize(obj, x0, method='L-BFGS-B', jac=grad, options={'maxiter': 300, 'ftol': 1e-12})
        return res.x.reshape(-1, 2), res.fun

    # Main loop: Increase target_r
    # Start with a small r
    target_r = 0.04
    step_size = 0.003
    current_points = points.copy()
    consecutive_failures = 0
    max_consecutive_failures = 5

    while time.time() - start_time < time_limit:
        # Try to optimize for target_r
        optimized_points, energy_val = optimize(current_points, target_r)
        
        # Check if we achieved a valid packing for this target_r
        # If energy is very small, it's valid.
        if energy_val < 1e-6:
            # Calculate actual radius
            actual_r = calc_radius(optimized_points)
            if actual_r > best_r:
                best_r = actual_r
                best_points = optimized_points.copy()
                current_points = optimized_points.copy()
                # Increase target_r
                target_r += step_size
                consecutive_failures = 0
            else:
                # If actual_r is not greater than best_r, but energy is low,
                # it means we are at a local minimum for this target_r but didn't improve global best.
                # Just increase target_r slightly to force expansion.
                target_r += step_size * 0.5
                consecutive_failures = 0
        else:
            # Optimization failed to find a valid configuration for target_r
            consecutive_failures += 1
            
            if consecutive_failures >= max_consecutive_failures:
                # Reduce target_r and restart from best_points
                target_r = max(0.03, best_r * 0.95)
                current_points = best_points + np.random.normal(0, 0.005, size=best_points.shape)
                # Pull back if outside
                norms = np.linalg.norm(current_points, axis=1)
                mask = norms > (1.0 - best_r)
                if np.any(mask):
                    current_points[mask] = best_points[mask]
                consecutive_failures = 0
            else:
                # Perturb and retry with same target_r
                current_points = best_points + np.random.normal(0, 0.01, size=best_points.shape)
                # Pull back if outside
                norms = np.linalg.norm(current_points, axis=1)
                mask = norms > (1.0 - best_r)
                if np.any(mask):
                    current_points[mask] = best_points[mask]

    return best_points

# EVOLVE-BLOCK-END
