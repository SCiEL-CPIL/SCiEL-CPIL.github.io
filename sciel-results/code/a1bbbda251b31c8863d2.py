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
    2. Lloyd's relaxation on a test grid.
    3. Precise convex-hull covering radius evaluation.
    4. Targeted Delaunay-based geometric refinement: move vertices of worst
       facets along geodesic bisectors to minimize the specific facet's
       circumradius, with a smooth surrogate (soft-max over facet radii).
    5. Final high-precision polish pass.
    6. Explicit deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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

    def compute_facet_gradients(points, facet_data, top_k=10):
        """
        Compute gradient of the covering radius (soft-max over top-k worst facets)
        with respect to the point positions. For each worst facet, the gradient
        direction for each vertex points away from the circumcenter (tangent).
        """
        n_points = len(points)
        gradient = np.zeros((n_points, 3))

        if not facet_data:
            return gradient

        # Use top-k worst facets with soft weighting
        sorted_facets = sorted(facet_data, key=lambda x: x[0], reverse=True)
        top_facets = sorted_facets[:top_k]

        # Soft-max weighting: weight_i = exp(r_i / T) / sum(exp(r_j / T))
        T = 0.02  # temperature in radians
        radii = np.array([f[0] for f in top_facets])
        weights = np.exp(radii / T)
        weights = weights / weights.sum()

        for w, (radius, simplex, cc_norm) in zip(weights, top_facets):
            for idx in simplex:
                current = points[idx]
                # Direction from current point toward circumcenter
                to_cc = cc_norm - current
                # Project onto tangent plane at current
                radial_component = np.dot(to_cc, current) * current
                tangent_dir = to_cc - radial_component
                norm_td = np.linalg.norm(tangent_dir)
                if norm_td > 1e-10:
                    # We want to minimize the circumradius, so move away from circumcenter
                    # Actually, moving a vertex toward the circumcenter shrinks the triangle
                    # Let's think: if we move v1 toward cc, the triangle gets smaller
                    # The circumradius decreases. So we move toward cc.
                    gradient[idx] += w * tangent_dir / norm_td

        return gradient

    def refine_delaunay_local(points, num_steps=100, lr=0.01, top_k=8):
        """
        Targeted local search using Delaunay triangulation:
        - Compute all facet circumradii
        - Compute gradient of soft-max over top-k worst facets
        - Move points along gradient (projected to tangent)
        - Normalize
        """
        for step in range(num_steps):
            facet_data = get_all_facet_data(points)
            if not facet_data:
                break

            grad = compute_facet_gradients(points, facet_data, top_k=top_k)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break

            # Normalize gradient to unit norm for stable steps
            grad = grad / grad_norm

            # Move points
            new_points = points + lr * grad
            # Project back to unit sphere
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            new_points = new_points / norms

            # Check for improvement
            new_radius = max(f[0] for f in facet_data) if facet_data else 0
            new_radius_deg = np.degrees(new_radius)

            points = new_points

            # Adaptive learning rate
            if step % 20 == 0 and step > 0:
                # Reduce learning rate if not improving
                lr *= 0.9

        return points

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

    def high_precision_polish(points, num_iters=50, lr=0.002):
        """
        Final high-precision polish: use very small steps with the
        Delaunay-based gradient to fine-tune the configuration.
        """
        for i in range(num_iters):
            facet_data = get_all_facet_data(points)
            if not facet_data:
                break

            grad = compute_facet_gradients(points, facet_data, top_k=12)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break

            grad = grad / grad_norm
            new_points = points + lr * grad
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            new_points = new_points / norms

            # Verify improvement
            old_max = max(f[0] for f in facet_data)
            new_facets = get_all_facet_data(new_points)
            if new_facets:
                new_max = max(f[0] for f in new_facets)
                if new_max < old_max:
                    points = new_points
                else:
                    # Revert if worse
                    pass
            else:
                points = new_points

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
    time_per_init = (deadline - start_time - 20.0) / len(initializations)

    for init_idx, init_pts in enumerate(initializations):
        if time.time() >= deadline - 15.0:
            break

        points = init_pts.copy()
        init_deadline = start_time + (init_idx + 1) * time_per_init

        # Phase 1: Lloyd's algorithm (fast convergence)
        step_size = 0.8
        max_lloyd_iters = 200
        iters = 0
        while time.time() < min(init_deadline, start_time + time_per_init * 0.4) and iters < max_lloyd_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.97
            iters += 1

        # Phase 2: Fine Lloyd refinement
        step_size = 0.1
        max_fine_iters = 100
        iters = 0
        while time.time() < min(init_deadline, start_time + time_per_init * 0.7) and iters < max_fine_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.95
            iters += 1
            if step_size < 0.01:
                break

        # Phase 3: Delaunay-based targeted refinement
        if time.time() < init_deadline - 3.0:
            points = refine_delaunay_local(points, num_steps=80, lr=0.015, top_k=8)
            points = separate_close_points(points)

        # Phase 4: High-precision polish
        if time.time() < init_deadline - 1.0:
            points = high_precision_polish(points, num_iters=30, lr=0.003)

        # Evaluate
        fast_score = compute_covering_radius_sampled(points, num_test_points=20000)
        if fast_score < best_score + 0.3:
            score = compute_covering_radius_precise(points)
            if score < best_score:
                best_score = score
                best_points = points.copy()

    # Final refinement with remaining time
    if best_points is not None and time.time() < deadline - 20.0:
        points = best_points.copy()

        # Extended Delaunay refinement
        if time.time() < deadline - 15.0:
            points = refine_delaunay_local(points, num_steps=150, lr=0.008, top_k=10)
            points = separate_close_points(points)

        # High-precision polish
        if time.time() < deadline - 5.0:
            points = high_precision_polish(points, num_iters=60, lr=0.001)

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
