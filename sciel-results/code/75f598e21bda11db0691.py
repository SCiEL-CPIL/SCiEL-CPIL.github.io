"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def calculate_radius(points):
        """Calculate the maximum radius for a given set of points."""
        if len(points) < 2:
            return 1.0
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        return min(min_pair_dist / 2.0, min_bound_dist)

    def bottleneck_info(points):
        """Return (radius, is_pair, i, j_or_idx) describing the active constraint."""
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)

        if min_pair_dist / 2.0 < min_bound_dist:
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            return min_pair_dist / 2.0, True, idx[0], idx[1]
        else:
            idx = np.argmin(1.0 - r_i)
            return min_bound_dist, False, idx, -1

    def hex_lattice_init(N_points):
        """Generate a hexagonal lattice initialization."""
        s = np.sqrt(2.0 * np.pi / (3.0 * np.sqrt(3.0) * N_points))
        k = int(1.0 / s) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                rows.append([x, y])
        points = np.array(rows)
        dists = np.linalg.norm(points, axis=1)
        points = points[dists <= 1.0 - s / 2]
        if len(points) < N_points:
            s *= 0.9
            k = int(1.0 / s) + 3
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = s * (i + 0.5 * (j % 2))
                    y = s * j * np.sqrt(3.0) / 2.0
                    rows.append([x, y])
            points = np.array(rows)
            dists = np.linalg.norm(points, axis=1)
            points = points[dists <= 1.0 - s / 2]
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        return points[sorted_idx[:N_points]].copy()

    def anneal_and_polish(points, max_iters, time_limit):
        """Anneal with bottleneck targeting, then coordinate-descent polish."""
        points = points.copy()
        current_radius = calculate_radius(points)
        best_points = points.copy()
        best_radius = current_radius

        # Phase 1: Annealed bottleneck adjustment
        T0 = 0.005
        T_end = 0.0001
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            T = T0 * (T_end / T0) ** (it / max_iters)

            radius, is_pair, i, j = bottleneck_info(points)

            if is_pair:
                direction = points[i] - points[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    continue
                direction /= norm
                # Try pushing apart
                for step in [0.002, 0.001, 0.0005]:
                    new_points = points.copy()
                    new_points[i] += direction * step
                    new_points[j] -= direction * step
                    # Keep inside circle
                    r_norms = np.linalg.norm(new_points, axis=1)
                    mask = r_norms > 1.0
                    if np.any(mask):
                        new_points[mask] /= r_norms[mask, np.newaxis]
                    new_radius = calculate_radius(new_points)
                    if new_radius > current_radius:
                        points = new_points
                        current_radius = new_radius
                        if current_radius > best_radius:
                            best_radius = current_radius
                            best_points = points.copy()
                        break
                    elif np.random.random() < np.exp((new_radius - current_radius) / T):
                        points = new_points
                        current_radius = new_radius
                        break
            else:
                idx = i
                point = points[idx]
                norm = np.linalg.norm(point)
                if norm < 1e-12:
                    continue
                direction = -point / norm
                for step in [0.002, 0.001, 0.0005]:
                    new_points = points.copy()
                    new_points[idx] += direction * step
                    new_radius = calculate_radius(new_points)
                    if new_radius > current_radius:
                        points = new_points
                        current_radius = new_radius
                        if current_radius > best_radius:
                            best_radius = current_radius
                            best_points = points.copy()
                        break
                    elif np.random.random() < np.exp((new_radius - current_radius) / T):
                        points = new_points
                        current_radius = new_radius
                        break

        # Phase 2: Coordinate-descent polish (direct min-clearance line search)
        points = best_points.copy()
        current_radius = best_radius
        for _ in range(50):
            if time.time() > time_limit:
                break
            radius, is_pair, i, j = bottleneck_info(points)

            if is_pair:
                direction = points[i] - points[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    break
                direction /= norm
                # Line search: maximize min clearance along direction
                best_step = 0.0
                best_r = current_radius
                for step in np.linspace(0.0001, 0.01, 20):
                    new_points = points.copy()
                    new_points[i] += direction * step
                    new_points[j] -= direction * step
                    r_norms = np.linalg.norm(new_points, axis=1)
                    mask = r_norms > 1.0
                    if np.any(mask):
                        new_points[mask] /= r_norms[mask, np.newaxis]
                    new_radius = calculate_radius(new_points)
                    if new_radius > best_r:
                        best_r = new_radius
                        best_step = step
                if best_step > 0:
                    points[i] += direction * best_step
                    points[j] -= direction * best_step
                    r_norms = np.linalg.norm(points, axis=1)
                    mask = r_norms > 1.0
                    if np.any(mask):
                        points[mask] /= r_norms[mask, np.newaxis]
                    current_radius = calculate_radius(points)
            else:
                idx = i
                point = points[idx]
                norm = np.linalg.norm(point)
                if norm < 1e-12:
                    break
                direction = -point / norm
                best_step = 0.0
                best_r = current_radius
                for step in np.linspace(0.0001, 0.01, 20):
                    new_points = points.copy()
                    new_points[idx] += direction * step
                    new_radius = calculate_radius(new_points)
                    if new_radius > best_r:
                        best_r = new_radius
                        best_step = step
                if best_step > 0:
                    points[idx] += direction * best_step
                    current_radius = calculate_radius(points)

            if current_radius > best_radius:
                best_radius = current_radius
                best_points = points.copy()

        return best_points, best_radius

    def ring_model_score(params, ring_radii, ring_counts, ring_offsets):
        """
        Compute the negative radius (to be minimized) for a ring-based model.
        params: array of angular offsets for each point (flattened).
        """
        points = np.empty((N, 2))
        idx = 0
        for r, count, offset in zip(ring_radii, ring_counts, ring_offsets):
            if count == 0:
                continue
            angles = offset + 2 * np.pi * np.arange(count) / count + params[idx:idx+count]
            idx += count
            points[idx-count:idx, 0] = r * np.cos(angles)
            points[idx-count:idx, 1] = r * np.sin(angles)
        
        if idx != N:
            return 10.0  # Penalty for mismatch

        # Check validity
        if not np.all(np.isfinite(points)):
            return 10.0
        r_norms = np.linalg.norm(points, axis=1)
        if np.any(r_norms > 1.0):
            # Penalty for being outside
            return 10.0 + np.sum(np.maximum(0, r_norms - 1.0))

        r = calculate_radius(points)
        return -r

    def optimize_rings(ring_radii, ring_counts, ring_offsets, time_limit):
        """
        Optimize the angular offsets for a given ring structure using Differential Evolution.
        """
        # Define bounds for angular offsets
        bounds = [(-0.1, 0.1)] * N  # Small perturbations around uniform distribution
        
        # Objective function wrapper
        def obj(params):
            return ring_model_score(params, ring_radii, ring_counts, ring_offsets)

        # Run differential evolution
        try:
            res = differential_evolution(
                obj,
                bounds,
                maxiter=100,
                popsize=10,
                tol=1e-6,
                seed=42,
                polish=False,
                update_interval=1,
                callback=None
            )
            # Reconstruct points from best params
            best_params = res.x
            points = np.empty((N, 2))
            idx = 0
            for r, count, offset in zip(ring_radii, ring_counts, ring_offsets):
                if count == 0:
                    continue
                angles = offset + 2 * np.pi * np.arange(count) / count + best_params[idx:idx+count]
                idx += count
                points[idx-count:idx, 0] = r * np.cos(angles)
                points[idx-count:idx, 1] = r * np.sin(angles)
            
            # Ensure validity
            r_norms = np.linalg.norm(points, axis=1)
            mask = r_norms > 1.0
            if np.any(mask):
                points[mask] /= r_norms[mask, np.newaxis]
            
            return points
        except Exception:
            return None

    # --- Main Execution ---

    best_overall = None
    best_overall_radius = 0.0

    # 1. Hex Lattice Start
    hex_pts = hex_lattice_init(N)
    t_limit = time.time() + 40.0
    pts, rad = anneal_and_polish(hex_pts, 500, t_limit)
    if rad > best_overall_radius:
        best_overall_radius = rad
        best_overall = pts.copy()

    if time.time() > deadline - 20.0:
        return best_overall if best_overall is not None else hex_pts

    # 2. Ring Model Optimization
    # Estimate number of rings
    # Approximate ring radii: r_k ~ k * sqrt(pi/N)
    # We'll try a few ring configurations
    
    s_est = np.sqrt(np.pi / N)
    
    # Configuration 1: Standard rings
    ring_radii = []
    ring_counts = []
    ring_offsets = []
    
    r = 0.0
    while len(ring_radii) < 20:
        r += s_est * 0.95
        if r > 1.0 - s_est / 2:
            break
        count = max(1, int(2 * np.pi * r / (s_est * 0.95)))
        ring_radii.append(r)
        ring_counts.append(count)
        ring_offsets.append(0.0)
        
    total_pts = sum(ring_counts)
    if total_pts < N:
        # Add more rings or adjust
        while total_pts < N:
            r += s_est * 0.5
            if r > 1.0 - s_est / 2:
                break
            count = max(1, int(2 * np.pi * r / (s_est * 0.95)))
            ring_radii.append(r)
            ring_counts.append(count)
            ring_offsets.append(0.0)
            total_pts += count
            
    # Trim or pad to exactly N
    # We need exactly N points. Let's adjust the last ring.
    diff = N - sum(ring_counts)
    if diff != 0 and len(ring_counts) > 0:
        ring_counts[-1] += diff
        if ring_counts[-1] < 1:
            # Invalid config, skip
            pass
        else:
            # Run optimization
            t_limit = time.time() + 50.0
            ring_pts = optimize_rings(ring_radii, ring_counts, ring_offsets, t_limit)
            if ring_pts is not None:
                rad = calculate_radius(ring_pts)
                if rad > best_overall_radius:
                    best_overall_radius = rad
                    best_overall = ring_pts.copy()

    if time.time() > deadline - 10.0:
        return best_overall if best_overall is not None else hex_pts

    # 3. L-BFGS-B Optimization on Full Coordinates (Inspiration 2)
    # Use the best point found so far as initialization
    if best_overall is not None:
        current_points = best_overall.copy()
        current_score = best_overall_radius
        
        def energy_and_grad(flat_x):
            x = flat_x.reshape((N, 2))
            
            # Calculate pairwise distances
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Avoid division by zero
            dist_sq_safe = dist_sq + 1e-9
            np.fill_diagonal(dist_sq_safe, 1.0)
            
            # Energy term 1: Pairwise repulsion
            inv_dist_sq = 1.0 / dist_sq_safe
            np.fill_diagonal(inv_dist_sq, 0.0)
            E_pair = np.sum(inv_dist_sq)
            
            # Gradient of E_pair
            grad_factor = -2.0 / (dist_sq_safe**2)
            np.fill_diagonal(grad_factor, 0.0)
            grad_pair = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
            
            # Energy term 2: Boundary repulsion
            r_i = np.linalg.norm(x, axis=1)
            r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
            d_b = 1.0 - r_i
            
            d_b_safe = np.where(d_b < 1e-6, 1e-6, d_b)
            
            E_bound = np.sum(1.0 / d_b_safe**2)
            
            # Gradient of E_bound
            grad_bound = 2.0 * (x / r_safe[:, np.newaxis]) / d_b_safe[:, np.newaxis]**3
            
            total_energy = E_pair + E_bound
            total_grad = grad_pair + grad_bound
            
            return total_energy, total_grad

        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(
                energy_and_grad, 
                current_points.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            optimized_points = res.x.reshape((N, 2))
            
            # Check validity
            if np.all(np.isfinite(optimized_points)):
                r_i = np.linalg.norm(optimized_points, axis=1)
                if np.all(r_i <= 1.0):
                    score = calculate_radius(optimized_points)
                    if score > best_overall_radius:
                        best_overall_radius = score
                        best_overall = optimized_points.copy()
        except Exception:
            pass

    # 4. Final Polish
    if best_overall is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 3.0
        if remaining > 5.0:
            max_iters = max(50, int(remaining * 10))
            pts, rad = anneal_and_polish(best_overall, max_iters, time.time() + remaining)
            if rad > best_overall_radius:
                best_overall_radius = rad
                best_overall = pts

    # Ensure validity
    if best_overall is None:
        best_overall = hex_lattice_init(N)

    r_norms = np.linalg.norm(best_overall, axis=1)
    mask = r_norms > 1.0
    if np.any(mask):
        best_overall[mask] /= r_norms[mask, np.newaxis]

    return best_overall
# EVOLVE-BLOCK-END
