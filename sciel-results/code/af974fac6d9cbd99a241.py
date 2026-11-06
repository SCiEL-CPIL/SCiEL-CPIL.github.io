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
    1. Fibonacci spiral initialization.
    2. Fast vectorized Lloyd's relaxation.
    3. Smooth gradient-based refinement using log-sum-exp over facet circumradii.
    4. Multiple restarts with perturbations to escape local minima.
    5. Explicit deadline guard for safe return.
    """
    start_time = time.time()
    deadline = start_time + 112.0

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        n = np.linalg.norm(pts, axis=1, keepdims=True)
        n = np.where(n < 1e-14, 1.0, n)
        return pts / n

    def compute_covering_radius_sampled(points, num_test=200000):
        test = fibonacci_sphere(num_test)
        dots = np.clip(test @ points.T, -1.0, 1.0)
        min_dot = np.min(dots, axis=1)
        return np.degrees(np.arccos(np.max(min_dot)))

    def compute_covering_radius_precise(points):
        """Exact covering radius via convex hull facet circumradii."""
        try:
            hull = ConvexHull(points)
        except Exception:
            return compute_covering_radius_sampled(points, 300000)

        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2*np.dot(e1,e1), 2*np.dot(e1,e2)],
                          [2*np.dot(e1,e2), 2*np.dot(e2,e2)]])
            b = np.array([np.dot(e1,e1), np.dot(e2,e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0]*e1 + params[1]*e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            if ang > max_rad:
                max_rad = ang
        return np.degrees(max_rad)

    def lloyd_iteration(points, test_points, step_size=1.0):
        dots = np.clip(test_points @ points.T, -1.0, 1.0)
        nearest = np.argmax(dots, axis=1)
        n_pts = len(points)
        sums = np.zeros((n_pts, 3))
        np.add.at(sums, nearest, test_points)
        counts = np.bincount(nearest, minlength=n_pts)
        counts_safe = np.where(counts == 0, 1, counts)
        centroids = sums / counts_safe[:, np.newaxis]
        centroids = normalize(centroids)
        empty = counts == 0
        if np.any(empty):
            centroids[empty] = points[empty]
        result = (1 - step_size) * points + step_size * centroids
        return normalize(result)

    def get_all_facet_data(points):
        """Return list of (angular_radius, simplex_indices, circumcenter) for all facets."""
        try:
            hull = ConvexHull(points)
        except Exception:
            return []
        data = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2*np.dot(e1,e1), 2*np.dot(e1,e2)],
                          [2*np.dot(e1,e2), 2*np.dot(e2,e2)]])
            b = np.array([np.dot(e1,e1), np.dot(e2,e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0]*e1 + params[1]*e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            data.append((ang, tuple(int(x) for x in simplex), cc))
        return data

    def refine_smooth_gradient(points, num_rounds=80, beta=40.0, lr=0.02):
        """
        Smooth gradient-based refinement using log-sum-exp over facet circumradii.
        This avoids the discontinuity of top-k selection and provides a smoother
        optimization landscape for escaping local minima.
        """
        pts = points.copy()
        current_lr = lr
        
        for rnd in range(num_rounds):
            if time.time() > deadline:
                break
            
            facet_data = get_all_facet_data(pts)
            if not facet_data:
                break
            
            # Extract angles and circumcenters
            angles = np.array([f[0] for f in facet_data])
            circumcenters = np.array([f[2] for f in facet_data])
            simplices = [f[1] for f in facet_data]
            
            if len(angles) == 0:
                break
            
            # Compute log-sum-exp weights for smooth max
            max_angle = np.max(angles)
            if max_angle < 1e-10:
                break
                
            exp_vals = np.exp(beta * (angles - max_angle))
            sum_exp = np.sum(exp_vals)
            if sum_exp < 1e-15:
                break
            weights = exp_vals / sum_exp
            
            # Compute gradient: sum over facets of weight * (tangent direction toward circumcenter)
            grad = np.zeros_like(pts)
            for w, simplex, cc in zip(weights, simplices, circumcenters):
                if w < 1e-8:
                    continue
                for idx in simplex:
                    cur = pts[idx]
                    to_cc = cc - cur
                    radial = np.dot(to_cc, cur) * cur
                    tangent = to_cc - radial
                    nt = np.linalg.norm(tangent)
                    if nt > 1e-12:
                        grad[idx] += w * (tangent / nt)
            
            gn = np.linalg.norm(grad)
            if gn > 1e-12:
                grad = grad / gn
                pts = pts + current_lr * grad
                pts = normalize(pts)
            
            # Adaptively reduce learning rate
            current_lr *= 0.96
            if current_lr < 0.001:
                current_lr = 0.001

        return pts

    def separate_close_points(points, min_cos=0.9995):
        """Push apart points that are nearly coincident."""
        pts = points.copy()
        for i in range(N):
            for j in range(i+1, N):
                d = np.dot(pts[i], pts[j])
                if d > min_cos:
                    diff = pts[i] - pts[j]
                    nd = np.linalg.norm(diff)
                    if nd > 1e-10:
                        diff = diff / nd
                        push = 0.005 * (1.0 - d) / (1.0 - min_cos + 1e-9)
                        pts[i] = pts[i] + push * diff
                        pts[j] = pts[j] - push * diff
                        pts[i] = pts[i] / np.linalg.norm(pts[i])
                        pts[j] = pts[j] / np.linalg.norm(pts[j])
        return pts

    # Pre-compute Lloyd test grid
    lloyd_test = fibonacci_sphere(150000)
    rng = np.random.RandomState(12345)

    best_points = None
    best_score = float('inf')

    num_restarts = 0
    max_restarts = 25

    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1

        # Initialize
        if num_restarts == 1:
            points = fibonacci_sphere(N)
        elif num_restarts == 2:
            # Slightly perturbed Fibonacci
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.03
            points = normalize(points)
        else:
            if best_points is not None and rng.random() < 0.6:
                points = best_points + rng.standard_normal((N, 3)) * 0.04
            else:
                points = rng.standard_normal((N, 3))
            points = normalize(points)

        # Phase 1: Lloyd's relaxation
        step_size = 0.6
        for it in range(80):
            if time.time() > deadline:
                break
            points = lloyd_iteration(points, lloyd_test, step_size)
            step_size *= 0.96
            if step_size < 0.02:
                step_size = 0.02

        # Phase 2: Separate close points
        points = separate_close_points(points)

        # Phase 3: Smooth gradient refinement (main refinement)
        points = refine_smooth_gradient(points, num_rounds=60, beta=40.0, lr=0.02)

        # Evaluate
        score = compute_covering_radius_precise(points)

        if score < best_score:
            best_score = score
            best_points = points.copy()

        # Phase 4: If good, do a second, finer refinement pass with higher beta
        if best_score < 18.5 and time.time() < deadline - 15:
            points = refine_smooth_gradient(points, num_rounds=50, beta=80.0, lr=0.008)
            points = separate_close_points(points, min_cos=0.9999)
            score2 = compute_covering_radius_precise(points)
            if score2 < best_score:
                best_score = score2
                best_points = points.copy()

    # Final normalization
    if best_points is not None:
        best_points = normalize(best_points)

    return best_points
# EVOLVE-BLOCK-END
