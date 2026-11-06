"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. Multiple diverse initializations.
    2. Lloyd's relaxation on a test grid for coarse positioning.
    3. Molecular dynamics with repulsive potential derived from log-distance
       to nearest neighbors, driving equalization of local density and covering radius.
    4. Final convex-hull based evaluation and targeted polish.
    5. Explicit deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_precise(points):
        """
        Compute the covering radius using the convex hull approach.
        For each triangular facet, compute the circumcenter and check
        if it lies on the sphere.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return compute_covering_radius_sampled(points, 200000)

        max_circumradius = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]

            e1 = v2 - v1
            e2 = v3 - v1

            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])

            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                continue

            circumcenter = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(circumcenter)
            if norm_cc < 1e-10:
                continue
            circumcenter_norm = circumcenter / norm_cc

            dot_val = np.clip(np.dot(circumcenter_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)

            if angular_radius > max_circumradius:
                max_circumradius = angular_radius

        return np.degrees(max_circumradius)

    def compute_covering_radius_sampled(points, num_test_points=200000):
        """
        Estimate covering radius by sampling many points on the sphere.
        """
        test_points = fibonacci_sphere(num_test_points)
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        min_dot = np.min(dot_products, axis=1)
        angles = np.arccos(min_dot)
        max_angle = np.max(angles)
        return np.degrees(max_angle)

    def lloyd_iteration(points, test_points, step_size=1.0):
        """One iteration of Lloyd's algorithm (vectorized)."""
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        nearest = np.argmax(dot_products, axis=1)

        n_points = len(points)
        sums = np.zeros((n_points, 3))
        np.add.at(sums, nearest, test_points)

        counts = np.bincount(nearest, minlength=n_points)
        counts_safe = np.where(counts == 0, 1, counts)

        centroids = sums / counts_safe[:, np.newaxis]

        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_points = centroids / norms

        empty_mask = counts == 0
        if np.any(empty_mask):
            new_points[empty_mask] = points[empty_mask]

        result = (1 - step_size) * points + step_size * new_points
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return result / norms

    def compute_repulsive_forces(points, temperature=0.1, k_repel=1.0, k_attr=0.5):
        """
        Compute molecular dynamics forces on the sphere.
        
        The potential is based on the logarithm of the distance to the nearest
        neighbor, which naturally drives points to equalize local density.
        
        Force on point i:
        - Repulsive: push away from nearby points (prevents clustering)
        - Attractive: pull toward the "center of mass" of neighbors (fills gaps)
        - Tangential projection: ensure forces are on the sphere surface
        
        The log-distance potential U(d) = -log(d) creates a force that is strong
        when points are close (repulsion) and weak when far (mild attraction via
        the global density balance).
        """
        n = len(points)
        forces = np.zeros((n, 3))
        
        # Compute pairwise distances (angular)
        dots = points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # Angular distances
        
        # Set diagonal to 1 to avoid self-interaction issues
        np.fill_diagonal(dists, 1.0)
        
        # Repulsive force: F_repel = k_repel * (1/d^2) * direction_away
        # Using log-distance derivative: d(-log(d))/dd = -1/d
        # Force magnitude ~ 1/d for log potential, but we use 1/d^2 for stronger separation
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                d = dists[i, j]
                if d < 1e-10:
                    continue
                
                # Direction from i to j
                dir_ij = points[j] - points[i]
                dir_norm = np.linalg.norm(dir_ij)
                if dir_norm < 1e-10:
                    continue
                dir_ij = dir_ij / dir_norm
                
                # Log-distance potential force: strong repulsion at short range
                # F = k_repel * exp(-d/temperature) / d
                repel_mag = k_repel * np.exp(-d / temperature) / max(d, 0.01)
                
                # Attractive force: weak, to fill gaps
                # Points that are far apart attract slightly to balance density
                attract_mag = k_attr * d / (d + 0.5)
                
                # Net force: repulsion dominates at short range, attraction at long range
                net_mag = repel_mag - attract_mag
                
                # Project force onto tangent plane at point i
                radial_comp = np.dot(dir_ij, points[i]) * points[i]
                tangent_force = net_mag * (dir_ij - radial_comp)
                
                forces[i] += tangent_force
        
        # Normalize forces to prevent numerical instability
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces = forces / max_force
        
        return forces

    def molecular_dynamics(points, num_steps=500, dt=0.01, temperature_start=0.3,
                           temperature_end=0.05, k_repel=1.0, k_attr=0.3):
        """
        Run molecular dynamics simulation on the sphere.
        
        Uses velocity Verlet integration with:
        - Repulsive potential from log-distance to nearest neighbors
        - Temperature annealing for convergence
        - Tangential constraint to keep points on sphere
        """
        n = len(points)
        velocities = np.zeros((n, 3))
        best_points = points.copy()
        best_score = np.inf
        
        for step in range(num_steps):
            # Anneal temperature
            t_frac = step / max(num_steps - 1, 1)
            temperature = temperature_start + (temperature_end - temperature_start) * t_frac
            
            # Compute forces
            forces = compute_repulsive_forces(points, temperature=temperature,
                                               k_repel=k_repel, k_attr=k_attr)
            
            # Velocity Verlet integration
            velocities = 0.99 * velocities + dt * forces  # Damping for stability
            
            # Update positions
            new_points = points + dt * velocities
            
            # Project back to unit sphere
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            new_points = new_points / norms
            
            # Remove radial velocity component (keep only tangential)
            for i in range(n):
                radial_vel = np.dot(velocities[i], new_points[i]) * new_points[i]
                velocities[i] = velocities[i] - radial_vel
            
            points = new_points
            
            # Periodic evaluation
            if step % 50 == 0 or step == num_steps - 1:
                score = compute_covering_radius_sampled(points, num_test_points=10000)
                if score < best_score:
                    best_score = score
                    best_points = points.copy()
        
        return best_points

    def get_all_facet_data(points):
        """
        Return list of (angular_radius, simplex_tuple, circumcenter_norm)
        for all facets of the convex hull.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return []

        facet_data = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]

            e1 = v2 - v1
            e2 = v3 - v1

            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])

            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                continue

            cc = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(cc)
            if norm_cc < 1e-10:
                continue
            cc_norm = cc / norm_cc

            dot_val = np.clip(np.dot(cc_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)

            facet_data.append((angular_radius, tuple(simplex), cc_norm))

        return facet_data

    def separate_close_points(points, min_angle=0.05):
        """Separate points that are too close together."""
        cos_min = np.cos(min_angle)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dot_val = np.dot(points[i], points[j])
                if dot_val > cos_min:
                    diff = points[i] - points[j]
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff > 1e-10:
                        diff_norm = diff / norm_diff
                        push = 0.01 * (1 - dot_val) / (1 - cos_min)
                        points[i] = points[i] + push * diff_norm
                        points[j] = points[j] - push * diff_norm
                        points[i] = points[i] / np.linalg.norm(points[i])
                        points[j] = points[j] / np.linalg.norm(points[j])
        return points

    # Pre-generate test points for Lloyd's
    num_test = 50000
    test_points = fibonacci_sphere(num_test)
    rng = np.random.RandomState(42)

    # Multiple initializations
    best_points = None
    best_score = np.inf

    initializations = []

    # 1. Fibonacci
    initializations.append(fibonacci_sphere(N))

    # 2. Random
    for _ in range(8):
        random_pts = rng.randn(N, 3)
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        initializations.append(random_pts)

    # 3. Grid-like (lat/lon)
    latitudes = np.linspace(-np.pi/2, np.pi/2, 10)
    longitudes = np.linspace(0, 2*np.pi, 5, endpoint=False)
    grid_pts = []
    for lat in latitudes:
        for lon in longitudes:
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts[:N])
    grid_pts /= np.linalg.norm(grid_pts, axis=1, keepdims=True)
    initializations.append(grid_pts)

    # 4. Perturbed Fibonacci
    fib = fibonacci_sphere(N)
    for _ in range(4):
        perturbed = fib + rng.randn(N, 3) * 0.1
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        initializations.append(perturbed)

    # Optimize each initialization with time budget
    time_per_init = (deadline - start_time - 25.0) / len(initializations)

    for init_idx, init_pts in enumerate(initializations):
        if time.time() >= deadline - 20.0:
            break

        points = init_pts.copy()
        init_deadline = start_time + (init_idx + 1) * time_per_init

        # Phase 1: Lloyd's algorithm (fast convergence)
        step_size = 0.8
        max_lloyd_iters = 150
        iters = 0
        while time.time() < min(init_deadline, start_time + time_per_init * 0.3) and iters < max_lloyd_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.97
            iters += 1

        # Phase 2: Fine Lloyd refinement
        step_size = 0.1
        max_fine_iters = 80
        iters = 0
        while time.time() < min(init_deadline, start_time + time_per_init * 0.5) and iters < max_fine_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.95
            iters += 1
            if step_size < 0.01:
                break

        # Phase 3: Molecular dynamics with repulsive potential
        if time.time() < init_deadline - 10.0:
            md_points = molecular_dynamics(
                points, 
                num_steps=300, 
                dt=0.02, 
                temperature_start=0.2, 
                temperature_end=0.05,
                k_repel=1.0, 
                k_attr=0.3
            )
            # Compare and keep better
            md_score = compute_covering_radius_sampled(md_points, num_test_points=10000)
            lloyd_score = compute_covering_radius_sampled(points, num_test_points=10000)
            if md_score < lloyd_score:
                points = md_points
            points = separate_close_points(points)

        # Phase 4: Final Lloyd polish
        if time.time() < init_deadline - 3.0:
            step_size = 0.05
            max_polish_iters = 50
            iters = 0
            while time.time() < init_deadline - 2.0 and iters < max_polish_iters:
                points = lloyd_iteration(points, test_points, step_size=step_size)
                step_size *= 0.93
                iters += 1
                if step_size < 0.005:
                    break

        # Evaluate
        fast_score = compute_covering_radius_sampled(points, num_test_points=20000)
        if fast_score < best_score + 0.3:
            score = compute_covering_radius_precise(points)
            if score < best_score:
                best_score = score
                best_points = points.copy()

    # Final refinement with remaining time
    if best_points is not None and time.time() < deadline - 30.0:
        points = best_points.copy()

        # Extended molecular dynamics
        if time.time() < deadline - 20.0:
            md_points = molecular_dynamics(
                points,
                num_steps=400,
                dt=0.015,
                temperature_start=0.1,
                temperature_end=0.02,
                k_repel=1.2,
                k_attr=0.2
            )
            md_score = compute_covering_radius_sampled(md_points, num_test_points=10000)
            if md_score < best_score:
                points = md_points
            points = separate_close_points(points)

        # Final Lloyd polish
        if time.time() < deadline - 10.0:
            step_size = 0.03
            max_polish_iters = 100
            iters = 0
            while time.time() < deadline - 5.0 and iters < max_polish_iters:
                points = lloyd_iteration(points, test_points, step_size=step_size)
                step_size *= 0.95
                iters += 1
                if step_size < 0.002:
                    break

        # Evaluate
        score = compute_covering_radius_precise(points)
        if score < best_score:
            best_score = score
            best_points = points.copy()

    # Final normalization
    if best_points is None:
        best_points = fibonacci_sphere(N)

    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms

    return best_points
# EVOLVE-BLOCK-END
