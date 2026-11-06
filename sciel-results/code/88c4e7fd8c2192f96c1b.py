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
    1. Diverse initializations (Fibonacci, random, perturbed).
    2. Repulsion-based (Rosen) gradient descent on the covering radius.
    3. Worst-facet targeted refinement (push vertices away from circumcenter).
    4. Precise convex-hull covering radius evaluation.
    5. Explicit deadline guard to return best result before timeout.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_facet_circumradii(points):
        """
        Compute circumradius (angular) for each triangular facet of the convex hull.
        Returns array of (radius, simplex, circumcenter_unit) for all facets.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return np.array([])

        results = []
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
            cc_unit = cc / norm_cc

            # Check the circumcenter is on the correct side (empty cap)
            dot_val = np.clip(np.dot(cc_unit, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)

            results.append((angular_radius, tuple(simplex), cc_unit))

        return results

    def compute_covering_radius_precise(points):
        """Compute the exact covering radius using convex hull facets."""
        facets = compute_facet_circumradii(points)
        if len(facets) == 0:
            # Fallback: sample-based estimate
            test_points = fibonacci_sphere(100000)
            dot_products = test_points @ points.T
            dot_products = np.clip(dot_products, -1.0, 1.0)
            min_dot = np.min(dot_products, axis=1)
            angles = np.arccos(min_dot)
            return np.degrees(np.max(angles))

        max_radius = max(f[0] for f in facets)
        return np.degrees(max_radius)

    def compute_covering_radius_sampled(points, num_test_points=50000):
        """Fast estimate of covering radius via sampling."""
        test_points = fibonacci_sphere(num_test_points)
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        min_dot = np.min(dot_products, axis=1)
        angles = np.arccos(min_dot)
        return np.degrees(np.max(angles))

    def rosen_repulsion_step(points, test_points, step_size=1.0):
        """
        One step of repulsion-based optimization.
        For each test point, find nearest config point; if angular distance > threshold,
        push that config point away from the test point (tangent direction).
        This directly reduces the covering radius.
        """
        n_points = len(points)
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        nearest = np.argmax(dot_products, axis=1)

        # For each test point, compute the direction to push its nearest point away
        # The direction is from the nearest config point toward the test point,
        # projected onto the tangent plane at the config point.
        nearest_pts = points[nearest]  # (num_test, 3)
        
        # Direction from config point to test point
        direction = test_points - nearest_pts  # (num_test, 3)
        
        # Project onto tangent plane at nearest config point
        # tangent = direction - (direction . config) * config
        radial_comp = np.sum(direction * nearest_pts, axis=1, keepdims=True)  # (num_test, 1)
        tangent = direction - radial_comp * nearest_pts  # (num_test, 3)
        
        # Only use test points that are far from their nearest (angular distance > ~10 deg)
        min_dot = np.sum(test_points * nearest_pts, axis=1)
        min_dot = np.clip(min_dot, -1.0, 1.0)
        angles = np.arccos(min_dot)
        # Weight: only push when angular distance is significant
        weights = np.where(angles > 0.1, angles, 0.0)  # radians, use as weight
        
        # Accumulate gradient per config point
        gradient = np.zeros((n_points, 3))
        np.add.at(gradient, nearest, tangent * weights[:, np.newaxis])
        
        # Normalize gradient
        grad_norm = np.linalg.norm(gradient, axis=1, keepdims=True)
        grad_norm = np.where(grad_norm < 1e-10, 1.0, grad_norm)
        gradient = gradient / grad_norm
        
        # Update points
        new_points = points + step_size * gradient
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_points = new_points / norms
        
        return new_points

    def worst_facet_refinement(points, num_steps=40, lr=0.01):
        """
        Targeted local search: for the worst facets, push their vertices
        away from the facet's circumcenter (tangent direction) to reduce
        that facet's circumradius.
        """
        for step in range(num_steps):
            facets = compute_facet_circumradii(points)
            if len(facets) == 0:
                break
            
            # Sort by radius descending
            facets_sorted = sorted(facets, key=lambda x: x[0], reverse=True)
            top_k = min(5, len(facets_sorted))
            
            gradient = np.zeros_like(points)
            for radius, simplex, cc_unit in facets_sorted[:top_k]:
                # Push each vertex away from circumcenter (tangent direction)
                for idx in simplex:
                    current = points[idx]
                    # Direction from circumcenter to vertex (tangent to sphere at vertex)
                    to_vertex = current - cc_unit
                    radial_comp = np.dot(to_vertex, current) * current
                    tangent = to_vertex - radial_comp
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        gradient[idx] += tangent / norm_t
            
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < 1e-10:
                break
            
            # Adaptive learning rate
            effective_lr = lr * (1.0 - step / num_steps * 0.5)
            points = points + effective_lr * gradient
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            points = points / norms
        
        return points

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

    def separate_close_points(points, min_angle=0.04):
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
                        push = 0.005 * (1 - dot_val) / (1 - cos_min)
                        points[i] = points[i] + push * diff_norm
                        points[j] = points[j] - push * diff_norm
                        points[i] = points[i] / np.linalg.norm(points[i])
                        points[j] = points[j] / np.linalg.norm(points[j])
        return points

    # Pre-generate test points
    num_test = 80000
    test_points = fibonacci_sphere(num_test)
    rng = np.random.RandomState(12345)

    # Multiple diverse initializations
    best_points = None
    best_score = np.inf

    initializations = []

    # 1. Fibonacci
    initializations.append(fibonacci_sphere(N))

    # 2. Random initializations
    for _ in range(8):
        random_pts = rng.randn(N, 3)
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        initializations.append(random_pts)

    # 3. Perturbed Fibonacci
    fib = fibonacci_sphere(N)
    for _ in range(4):
        perturbed = fib + rng.randn(N, 3) * 0.15
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        initializations.append(perturbed)

    # 4. Grid-like
    latitudes = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, 10)
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

    # Optimize each initialization with time budget
    num_inits = len(initializations)
    time_per_init = (deadline - start_time - 20.0) / num_inits

    for init_idx, init_pts in enumerate(initializations):
        if time.time() >= deadline - 15.0:
            break

        points = init_pts.copy()
        init_deadline = start_time + (init_idx + 1) * time_per_init

        # Phase 1: Lloyd's algorithm for fast convergence
        step_size = 0.8
        max_lloyd_iters = 200
        iters = 0
        while time.time() < min(init_deadline, start_time + time_per_init * 0.4) and iters < max_lloyd_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.97
            iters += 1

        # Phase 2: Repulsion-based optimization (core driver)
        step_size = 0.05
        max_rep_iters = 300
        iters = 0
        prev_score = np.inf
        no_improve_count = 0
        while time.time() < min(init_deadline, start_time + time_per_init * 0.75) and iters < max_rep_iters:
            points = rosen_repulsion_step(points, test_points, step_size=step_size)
            step_size *= 0.98
            iters += 1
            
            # Periodic evaluation
            if iters % 20 == 0:
                current_score = compute_covering_radius_sampled(points, 20000)
                if current_score < prev_score - 0.01:
                    prev_score = current_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count > 5:
                    break

        # Phase 3: Worst-facet refinement
        if time.time() < init_deadline - 3.0:
            points = worst_facet_refinement(points, num_steps=30, lr=0.008)
            points = separate_close_points(points)

        # Evaluate
        fast_score = compute_covering_radius_sampled(points, num_test_points=30000)
        if fast_score < best_score + 0.3:
            score = compute_covering_radius_precise(points)
            if score < best_score:
                best_score = score
                best_points = points.copy()

    # Final refinement with remaining time
    if best_points is not None and time.time() < deadline - 20.0:
        points = best_points.copy()

        # Fine repulsion
        step_size = 0.02
        max_final_iters = 200
        iters = 0
        while time.time() < deadline - 25.0 and iters < max_final_iters:
            points = rosen_repulsion_step(points, test_points, step_size=step_size)
            step_size *= 0.97
            iters += 1

            if iters % 15 == 0:
                current_score = compute_covering_radius_precise(points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = points.copy()

        # Worst-facet refinement passes
        if time.time() < deadline - 10.0:
            points = best_points.copy()
            points = worst_facet_refinement(points, num_steps=30, lr=0.005)
            points = separate_close_points(points)
            current_score = compute_covering_radius_precise(points)
            if current_score < best_score:
                best_score = current_score
                best_points = points.copy()

        # Another pass
        if time.time() < deadline - 5.0:
            points = best_points.copy()
            step_size = 0.01
            for _ in range(50):
                if time.time() >= deadline - 5.0:
                    break
                points = rosen_repulsion_step(points, test_points, step_size=step_size)
                step_size *= 0.95
            points = worst_facet_refinement(points, num_steps=20, lr=0.003)
            points = separate_close_points(points)
            current_score = compute_covering_radius_precise(points)
            if current_score < best_score:
                best_score = current_score
                best_points = points.copy()

    # Final normalization
    if best_points is None:
        best_points = fibonacci_sphere(N)

    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms

    return best_points
# EVOLVE-BLOCK-END
