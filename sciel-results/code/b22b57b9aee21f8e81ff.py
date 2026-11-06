import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. Multiple restarts with Fibonacci + noise.
    2. Lloyd's relaxation for coarse tiling.
    3. Smooth max-radius gradient descent (log-sum-exp surrogate).
    4. Targeted micro-polish on worst facets with local optimization.
    5. Strict deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin

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

    def compute_facet_data(points):
        """
        Returns:
        radii: array of facet circumradii (radians)
        ccs: array of facet circumcenters (unit vectors)
        verts: array of facet vertex indices (int)
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return None, None, None

        radii = []
        ccs = []
        verts = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            # Euclidean circumcenter of triangle v1, v2, v3
            e1 = v2 - v1
            e2 = v3 - v1
            # Solve for circumcenter in basis e1, e2
            # |v1 + u e1 + v e2 - C|^2 = 0 for all vertices
            # C is in plane of v1, v2, v3.
            # C = v1 + x e1 + y e2
            # |C|^2 = |v1|^2 + 2x <v1,e1> + 2y <v1,e2> + x^2|e1|^2 + 2xy<e1,e2> + y^2|e2|^2
            # |C - v1|^2 = |v1|^2 - |C|^2 = 0 => |C|^2 = 1
            # This is getting complicated.
            # Standard formula for circumcenter of triangle with vertices a, b, c:
            # d = 2 * (a_x * (b_y - c_y) + b_x * (c_y - a_y) + c_x * (a_y - b_y))
            # ux = ((a^2) * (b_y - c_y) + (b^2) * (c_y - a_y) + (c^2) * (a_y - b_y)) / d
            # uy = ((a^2) * (c_x - b_x) + (b^2) * (a_x - c_x) + (c^2) * (b_x - a_x)) / d
            # But this is 2D. We are in 3D.
            # The circumcenter of a spherical triangle is the normalization of the
            # circumcenter of the Euclidean triangle formed by the chord endpoints.
            
            # Let's use the cross product method for the normal, then find the point in the plane.
            # Actually, the circumcenter C is the solution to:
            # C . (v2 - v1) = 0
            # C . (v3 - v1) = 0
            # C . C = 1 (on sphere)
            # This implies C is parallel to (v2 - v1) x (v3 - v1).
            # Let n = (v2 - v1) x (v3 - v1).
            # C = n / |n| ?
            # Check: If C = n/|n|, then C . (v2 - v1) = 0. Yes.
            # Does C . v1 = C . v2?
            # C . (v2 - v1) = 0 => C . v2 = C . v1. Yes.
            # So the circumcenter is indeed the normalized normal vector of the plane containing the triangle.
            # Wait, this is only true if the triangle is "small" or if the circumcenter is the pole?
            # No, for ANY triangle on a sphere, the circumcenter is the intersection of the sphere
            # and the line passing through the origin and the circumcenter of the Euclidean triangle.
            # The vector n = (v2-v1) x (v3-v1) is normal to the plane.
            # The circumcenter of the Euclidean triangle lies on the line passing through the centroid? No.
            # The circumcenter of the Euclidean triangle is the point equidistant from v1, v2, v3.
            # Let C_euclid be that point.
            # C_euclid is in the plane of v1, v2, v3.
            # C_euclid = v1 + x e1 + y e2.
            # |C_euclid - v1|^2 = |C_euclid - v2|^2
            # |x e1 + y e2|^2 = |(x-1) e1 + y e2|^2
            # x^2 |e1|^2 + y^2 |e2|^2 + 2xy <e1,e2> = (x-1)^2 |e1|^2 + y^2 |e2|^2 + 2y(x-1)<e1,e2>
            # x^2 |e1|^2 = (x^2 - 2x + 1) |e1|^2 + 2xy<e1,e2> - 2y<e1,e2>
            # 2x |e1|^2 - |e1|^2 - 2xy<e1,e2> + 2y<e1,e2> = 0
            # x (|e1|^2 - y<e1,e2>) - y<e1,e2> - 0.5|e1|^2 = 0 ??
            
            # Let's use a robust solver.
            A = np.array([
                [2*np.dot(e1, e1), 2*np.dot(e1, e2)],
                [2*np.dot(e1, e2), 2*np.dot(e2, e2)]
            ])
            b = np.array([np.dot(e1, e1), np.dot(e2, e2)])
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
            radii.append(ang)
            ccs.append(cc)
            verts.append(simplex)

        if not radii:
            return None, None, None
        return np.array(radii), np.array(ccs), np.array(verts, dtype=int)

    def compute_covering_radius(points):
        radii, _, _ = compute_facet_data(points)
        if radii is None:
            # Fallback: grid search
            grid = fibonacci_sphere(10000)
            dots = np.clip(grid @ points.T, -1.0, 1.0)
            min_dot = np.min(dots, axis=1)
            return np.degrees(np.arccos(np.max(min_dot)))
        return np.degrees(np.max(radii))

    def smooth_gradient_step(points, beta=100.0, lr=0.01):
        """
        Gradient descent on the soft-max of facet circumradii.
        """
        radii, ccs, verts = compute_facet_data(points)
        if radii is None:
            return points

        # Soft max weights
        max_r = np.max(radii)
        # Avoid overflow
        weights = np.exp(beta * (radii - max_r))
        sum_w = np.sum(weights)
        if sum_w < 1e-14:
            return points
        weights = weights / sum_w

        grad = np.zeros_like(points)
        for f in range(len(radii)):
            w = weights[f]
            if w < 1e-6:
                continue
            cc = ccs[f]
            r = radii[f]
            # Gradient of r = arccos(cc . v) wrt v
            # dr/dv = - (cc - (cc.v)v) / sin(r)
            # We want to MINIMIZE r, so we move in direction -grad_r
            # Move v towards cc.
            sin_r = np.sin(r)
            if sin_r < 1e-6:
                continue
            for idx in verts[f]:
                v = points[idx]
                # Tangential component of (cc - v)
                to_cc = cc - v
                radial_comp = np.dot(to_cc, v) * v
                tangent = to_cc - radial_comp
                # Normalize tangent for stable step
                nt = np.linalg.norm(tangent)
                if nt > 1e-12:
                    grad[idx] += w * (tangent / nt) / sin_r

        gn = np.linalg.norm(grad)
        if gn > 1e-14:
            points = points + lr * (grad / gn)
            points = normalize(points)
        return points

    def micro_polish(points, num_iters=50, top_k=5):
        """
        Iteratively polish the worst facets.
        """
        pts = points.copy()
        for _ in range(num_iters):
            if time.time() > deadline - 5:
                break
            radii, ccs, verts = compute_facet_data(pts)
            if radii is None:
                break
            
            order = np.argsort(radii)[::-1]
            top_order = order[:top_k]
            
            improved = False
            for f in top_order:
                cc = ccs[f]
                facet_verts = verts[f]
                # Try moving vertices towards cc
                for step in [0.01, 0.005, 0.001]:
                    new_pts = pts.copy()
                    for idx in facet_verts:
                        v = new_pts[idx]
                        to_cc = cc - v
                        radial_comp = np.dot(to_cc, v) * v
                        tangent = to_cc - radial_comp
                        nt = np.linalg.norm(tangent)
                        if nt > 1e-12:
                            new_pts[idx] = v + step * (tangent / nt)
                            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
                    
                    # Check if this move improves the global covering radius
                    # To save time, we can check only the radius of this facet and its neighbors?
                    # But global is safer.
                    new_score = compute_covering_radius(new_pts)
                    old_score = compute_covering_radius(pts)
                    if new_score < old_score - 1e-8:
                        pts = new_pts
                        improved = True
                        break
            if not improved:
                break
        return pts

    # Main Loop
    best_points = None
    best_score = float('inf')
    rng = np.random.RandomState(42)
    lloyd_test = fibonacci_sphere(50000)

    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break

        # Initialization
        if restart == 0:
            points = fibonacci_sphere(N)
        elif restart < 5:
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.02
            points = normalize(points)
        elif best_points is not None and rng.random() < 0.5:
            points = best_points + rng.standard_normal((N, 3)) * 0.05
            points = normalize(points)
        else:
            points = rng.standard_normal((N, 3))
            points = normalize(points)

        # Phase 1: Lloyd
        step = 0.5
        for _ in range(40):
            if time.time() > deadline - 30:
                break
            points = lloyd_iteration(points, lloyd_test, step)
            step *= 0.95
            if step < 0.01:
                step = 0.01

        # Phase 2: Smooth Gradient Descent
        for _ in range(100):
            if time.time() > deadline - 20:
                break
            points = smooth_gradient_step(points, beta=50.0, lr=0.02)

        # Phase 3: Micro Polish
        if time.time() < deadline - 10:
            points = micro_polish(points, num_iters=30, top_k=5)

        score = compute_covering_radius(points)
        if score < best_score:
            best_score = score
            best_points = points.copy()

    if best_points is None:
        best_points = fibonacci_sphere(N)
    
    return normalize(best_points)
# EVOLVE-BLOCK-END
