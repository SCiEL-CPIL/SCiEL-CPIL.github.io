import time
import numpy as np
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def compute_radius(positions):
        """Compute the actual radius from positions."""
        from scipy.spatial.distance import cdist
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        xs = positions[:, 0]
        ys = positions[:, 1]
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)
        min_boundary_dist = np.min(dist_b)
        
        return min(min_pair_dist / 2, min_boundary_dist)

    def smooth_overlap_energy(positions, r, beta=50.0):
        """
        Smooth surrogate for the overlap energy using soft-min.
        Instead of max(0, 2r - d)^2, we use a smooth approximation.
        For pairs: soft penalty when d < 2r
        For boundaries: soft penalty when dist_b < r
        """
        from scipy.spatial.distance import cdist
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, 2*r + 1)
        
        xs = positions[:, 0]
        ys = positions[:, 1]
        
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)
        
        # Pairwise: smooth max(0, 2r - d)^2
        # Use softplus-like: (1/beta) * log(1 + exp(beta*(2r-d)))^2 approx
        # Simpler: use a quadratic barrier with smooth transition
        # f(x) = x^2 for x>0, smooth for x<0
        # Let g(x) = x^2 if x > 0, else (x^2 * exp(-beta*x)) / (1+exp(-beta*x))^2
        # Even simpler: use (2r - d)^2 * sigmoid(beta*(2r-d))
        pair_diff = 2*r - dists
        # Smooth step: sigmoid(beta * pair_diff) is 1 when pair_diff >> 0, 0 when pair_diff << 0
        # To avoid overflow, clip
        pair_diff_clipped = np.clip(pair_diff, -20/beta, 20/beta)
        sig = 1.0 / (1.0 + np.exp(-beta * pair_diff_clipped))
        pair_energy = np.sum((pair_diff**2) * sig)
        
        # Boundary: smooth max(0, r - dist_b)^2
        bound_diff = r - dist_b
        bound_diff_clipped = np.clip(bound_diff, -20/beta, 20/beta)
        sig_b = 1.0 / (1.0 + np.exp(-beta * bound_diff_clipped))
        boundary_energy = np.sum((bound_diff**2) * sig_b)
        
        return pair_energy + boundary_energy

    def hard_overlap_energy(positions, r):
        """Original hard overlap energy for feasibility checking."""
        from scipy.spatial.distance import cdist
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, 2*r + 1)
        
        xs = positions[:, 0]
        ys = positions[:, 1]
        
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)
        
        pair_overlap = np.maximum(0, 2*r - dists)
        pair_energy = np.sum(pair_overlap**2)
        
        boundary_overlap = np.maximum(0, r - dist_b)
        boundary_energy = np.sum(boundary_overlap**2)
        
        return pair_energy + boundary_energy

    def hexagonal_init(n, r_target=0.036):
        """Create a hexagonal grid centered in the square."""
        # Spacing based on target radius
        s = 2 * r_target * 1.001  # Slightly more than 2r to avoid overlaps
        points = []
        # Generate hex grid
        for i in range(-15, 15):
            for j in range(-15, 15):
                x = j * s + (i % 2) * s / 2
                y = i * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points)
        # Filter points in the square with some margin
        mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & \
               (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        points = points[mask]
        
        # If we have more than n points, select the best n
        if len(points) > n:
            # Score points by distance to center (prefer central points)
            dists_to_center = np.sum(points**2, axis=1)
            # Also consider boundary distance
            dist_x = np.minimum(0.5 - points[:, 0], points[:, 0] + 0.5)
            dist_y = np.minimum(0.5 - points[:, 1], points[:, 1] + 0.5)
            dist_b = np.minimum(dist_x, dist_y)
            # Prefer points that are central AND have good boundary distance
            score = dists_to_center - 0.1 * dist_b
            idx = np.argsort(score)[:n]
            points = points[idx]
        elif len(points) < n:
            needed = n - len(points)
            extra = np.random.uniform(-0.4, 0.4, size=(needed, 2))
            points = np.vstack([points, extra])
            
        return points

    def square_grid_init(n):
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]

    def polish(positions, r_target, max_iter=200, tol=1e-10):
        """High-precision polish pass with tight tolerances."""
        if time.time() > deadline - 10:
            return positions
            
        bounds = [(-0.5, -0.5), (0.5, 0.5)] * N
        
        try:
            result = minimize(
                lambda x: smooth_overlap_energy(x, r_target, beta=100.0),
                positions,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < 1e-8:
                return result.x
        except Exception:
            pass
        
        # Try hard energy polish
        try:
            result = minimize(
                lambda x: hard_overlap_energy(x, r_target),
                positions,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < 1e-8:
                return result.x
        except Exception:
            pass
        
        return positions

    best_radius = -np.inf
    best_positions = None
    
    # Multiple initializations with different random seeds
    np.random.seed(42)
    inits = [
        hexagonal_init(N, 0.036),
        hexagonal_init(N, 0.035),
        square_grid_init(N),
        np.random.uniform(-0.45, 0.45, size=(N, 2))
    ]
    
    for init_pos in inits:
        if time.time() > deadline - 20:
            break
            
        current_pos = init_pos.copy()
        
        # Start with a conservative radius and inflate
        r = 0.025
        step = 0.002
        
        while r < 0.042:
            if time.time() > deadline - 15:
                break
                
            # Optimize with smooth energy
            try:
                result = minimize(
                    lambda x: smooth_overlap_energy(x, r, beta=50.0),
                    current_pos,
                    method='L-BFGS-B',
                    bounds=[(-0.5, -0.5), (0.5, 0.5)] * N,
                    options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
                )
                current_pos = result.x
                energy = result.fun
            except Exception:
                break
            
            # Check feasibility with hard energy
            if time.time() > deadline - 10:
                break
                
            hard_energy = hard_overlap_energy(current_pos, r)
            
            if hard_energy < 1e-6:
                # Feasible! Try to increase r
                r += step
            else:
                # Infeasible. Binary search for the max feasible r
                r_low = r - step
                r_high = r
                
                for _ in range(12):
                    if time.time() > deadline - 10:
                        break
                    r_mid = (r_low + r_high) / 2
                    
                    try:
                        res_mid = minimize(
                            lambda x: smooth_overlap_energy(x, r_mid, beta=50.0),
                            current_pos,
                            method='L-BFGS-B',
                            bounds=[(-0.5, -0.5), (0.5, 0.5)] * N,
                            options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
                        )
                        current_pos = res_mid.x
                        energy_mid = hard_overlap_energy(current_pos, r_mid)
                        if energy_mid < 1e-6:
                            r_low = r_mid
                        else:
                            r_high = r_mid
                    except Exception:
                        break
                
                # r_low is the best feasible radius found
                if r_low > best_radius:
                    best_radius = r_low
                    best_positions = current_pos.copy()
                
                break  # Move to next initialization
        
        # If we reached the upper bound and were still feasible
        if r >= 0.042:
            try:
                hard_energy = hard_overlap_energy(current_pos, r)
                if hard_energy < 1e-6 and r > best_radius:
                    best_radius = r
                    best_positions = current_pos.copy()
            except Exception:
                pass

    # Fallback if no valid position found
    if best_positions is None:
        best_positions = square_grid_init(N)
    
    # Final high-precision polish
    if best_positions is not None and time.time() < deadline - 15:
        # Use the actual computed radius as target for polishing
        actual_r = compute_radius(best_positions)
        # Try to polish at slightly higher radius
        for r_try in [actual_r + 0.0001, actual_r + 0.00005, actual_r]:
            if time.time() > deadline - 10:
                break
            polished = polish(best_positions, r_try, max_iter=300, tol=1e-12)
            r_polished = compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_positions = polished
    
    # Final validation and clamping
    best_positions = np.clip(best_positions, -0.49999, 0.49999)
    
    return best_positions
