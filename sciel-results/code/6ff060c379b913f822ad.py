"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

TIME_LIMIT = 110.0  # safety margin before 120s hard kill


def _normalize(pts):
    """Normalize each row to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _compute_covering_radius(pts):
    """
    Compute the covering radius (in radians) as the max circumradius of facets
    of the convex hull, restricted to facets whose circumcenter lies in the
    outward normal direction (empty cap condition).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    max_r = 0.0
    for simplex in hull.simplices:
        tri = simplex[:3]  # take first 3 points (should be triangles in 3D)
        if len(tri) < 3:
            continue
        p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]

        # Check if it's a triangle (3 vertices)
        # Compute circumcenter of the triangle on the unit sphere
        # Using the formula: C = (p1 + p2 + p3) / |p1 + p2 + p3|
        C = p1 + p2 + p3
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C

        # Check that C is on the "outside" (outward normal direction)
        # The facet normal should point outward. For a convex hull, the normal
        # should be in the direction of C (the circumcenter of the cap)
        normal = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-10:
            continue
        normal = normal / norm_n

        # Check that the normal points in the same direction as C
        # (i.e., the cap is "empty" - the circumcenter is outside the convex hull)
        if np.dot(normal, C) < 0:
            # Flip normal
            normal = -normal
            if np.dot(normal, C) < 0:
                # C is not on the outward side, skip
                continue

        # Compute circumradius: angular distance from C to any vertex
        cos_r = np.clip(np.dot(C, p1), -1.0, 1.0)
        R_circ = np.arccos(cos_r)

        # Also check midpoint of edges to ensure the cap is correctly bounded
        # The actual covering radius contribution is the max angular distance
        # from the circumcenter to the three vertices, which is R_circ.
        # But we also need to verify the cap is empty (no other points inside).
        # For now, we use the circumradius of the facet.

        if R_circ > max_r:
            max_r = R_circ

    return max_r


def _smooth_covering_radius(pts, alpha=100.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp over
    angular distances to all points. This is differentiable and can be
    used for gradient-based optimization.
    """
    # Compute all pairwise angular distances
    # For each point on sphere (approximated by the N points), the covering radius
    # is the max over y of min_i angle(y, p_i)
    # We approximate by checking at the vertices of the convex hull facets
    # But for a smooth surrogate, we use a different approach:
    # Use the fact that for the covering problem, we can use a soft-max
    # over the circumradii of the Delaunay triangles (facets).

    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    radii = []
    for simplex in hull.simplices:
        tri = simplex[:3]
        if len(tri) < 3:
            continue
        p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]

        C = p1 + p2 + p3
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C

        normal = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-10:
            continue
        normal = normal / norm_n

        if np.dot(normal, C) < 0:
            normal = -normal
            if np.dot(normal, C) < 0:
                continue

        cos_r = np.clip(np.dot(C, p1), -1.0, 1.0)
        R_circ = np.arccos(cos_r)
        radii.append(R_circ)

    if not radii:
        return np.pi

    radii = np.array(radii)
    # Log-sum-exp for smooth max
    max_r = np.max(radii)
    if max_r < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(alpha * (radii - max_r)))) / alpha
    return max_r + log_sum


def _facet_based_local_search(pts, max_iter=200, base_step=0.05):
    """
    Local search that moves points to reduce the largest circumradius facet.
    """
    pts = _normalize(pts.copy())

    for iteration in range(max_iter):
        if time.time() > TIME_LIMIT - 15:
            break

        # Find the facet with the largest circumradius
        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        max_r = 0.0
        max_facet_idx = -1
        for i, simplex in enumerate(hull.simplices):
            tri = simplex[:3]
            if len(tri) < 3:
                continue
            p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]

            C = p1 + p2 + p3
            norm_C = np.linalg.norm(C)
            if norm_C < 1e-10:
                continue
            C = C / norm_C

            normal = np.cross(p2 - p1, p3 - p1)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-10:
                continue
            normal = normal / norm_n

            if np.dot(normal, C) < 0:
                normal = -normal
                if np.dot(normal, C) < 0:
                    continue

            cos_r = np.clip(np.dot(C, p1), -1.0, 1.0)
            R_circ = np.arccos(cos_r)

            if R_circ > max_r:
                max_r = R_circ
                max_facet_idx = i

        if max_facet_idx == -1:
            break

        tri = hull.simplices[max_facet_idx][:3]
        p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]

        # Compute circumcenter
        C = p1 + p2 + p3
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            break
        C = C / norm_C

        # Move each vertex away from the circumcenter to increase the cap
        step = base_step / (1.0 + 0.1 * iteration)
        for idx in tri:
            p = pts[idx]
            dir_vec = p - C
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec = dir_vec / norm_dir
                new_p = p + step * dir_vec
                new_p = new_p / np.linalg.norm(new_p)
                pts[idx] = new_p

    return _normalize(pts)


def _gradient_descent(pts, max_iter=500, lr=0.01):
    """
    Gradient-based optimization of the smooth covering radius surrogate.
    """
    pts = _normalize(pts.copy())

    best_score = _compute_covering_radius(pts)
    best_pts = pts.copy()

    for iteration in range(max_iter):
        if time.time() > TIME_LIMIT - 10:
            break

        # Numerical gradient of the smooth covering radius
        grad = np.zeros_like(pts)
        eps = 1e-5
        base_f = _smooth_covering_radius(pts, alpha=100.0)

        for i in range(N):
            for j in range(3):
                orig = pts[i, j]
                pts[i, j] = orig + eps
                pts[i] = pts[i] / np.linalg.norm(pts[i])
                f_plus = _smooth_covering_radius(pts, alpha=100.0)
                pts[i, j] = orig
                pts[i] = pts[i] / np.linalg.norm(pts[i])

                pts[i, j] = orig - eps
                pts[i] = pts[i] / np.linalg.norm(pts[i])
                f_minus = _smooth_covering_radius(pts, alpha=100.0)
                pts[i, j] = orig
                pts[i] = pts[i] / np.linalg.norm(pts[i])

                grad[i, j] = (f_plus - f_minus) / (2 * eps)

        # Project gradient onto tangent space and update
        for i in range(N):
            # Tangent space projection: g - (g·p)*p
            g_dot_p = np.dot(grad[i], pts[i])
            grad_tangent = grad[i] - g_dot_p * pts[i]
            new_p = pts[i] - lr * grad_tangent
            new_p = new_p / np.linalg.norm(new_p)
            pts[i] = new_p

        # Check improvement
        score = _compute_covering_radius(pts)
        if score < best_score - 1e-8:
            best_score = score
            best_pts = pts.copy()
        elif iteration > 50:
            # Reset to best if stuck
            pts = best_pts.copy()
            lr *= 0.9

    return best_pts


def _polish(pts, max_iter=300):
    """
    Final high-precision polish using very small step local search.
    """
    pts = _normalize(pts.copy())

    for iteration in range(max_iter):
        if time.time() > TIME_LIMIT - 5:
            break

        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        max_r = 0.0
        max_facet_idx = -1
        for i, simplex in enumerate(hull.simplices):
            tri = simplex[:3]
            if len(tri) < 3:
                continue
            p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]

            C = p1 + p2 + p3
            norm_C = np.linalg.norm(C)
            if norm_C < 1e-10:
                continue
            C = C / norm_C

            normal = np.cross(p2 - p1, p3 - p1)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-10:
                continue
            normal = normal / norm_n

            if np.dot(normal, C) < 0:
                normal = -normal
                if np.dot(normal, C) < 0:
                    continue

            cos_r = np.clip(np.dot(C, p1), -1.0, 1.0)
            R_circ = np.arccos(cos_r)

            if R_circ > max_r:
                max_r = R_circ
                max_facet_idx = i

        if max_facet_idx == -1:
            break

        tri = hull.simplices[max_facet_idx][:3]
        p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]

        C = p1 + p2 + p3
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            break
        C = C / norm_C

        step = 0.005 / (1.0 + 0.05 * iteration)
        for idx in tri:
            p = pts[idx]
            dir_vec = p - C
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec = dir_vec / norm_dir
                new_p = p + step * dir_vec
                new_p = new_p / np.linalg.norm(new_p)
                pts[idx] = new_p

    return _normalize(pts)


def _generate_initial_configurations():
    """Generate multiple diverse initial configurations."""
    configs = []

    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib)

    # 2. Random configurations
    np.random.seed(42)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        np.random.seed(seed)
        pts = np.random.randn(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)

    return configs


def construct():
    start_time = time.time()

    configs = _generate_initial_configurations()

    best_score = np.pi
    best_points = configs[0].copy()

    for init in configs:
        if time.time() > TIME_LIMIT - 20:
            break

        # Phase 1: Facet-based local search
        pts = _facet_based_local_search(init, max_iter=100, base_step=0.05)

        # Phase 2: Gradient descent polish
        if time.time() > TIME_LIMIT - 15:
            score = _compute_covering_radius(pts)
            if score < best_score:
                best_score = score
                best_points = pts.copy()
            continue

        pts_grad = _gradient_descent(pts, max_iter=100, lr=0.005)

        # Phase 3: Final polish
        pts_final = _polish(pts_grad, max_iter=200)

        score = _compute_covering_radius(pts_final)
        if score < best_score:
            best_score = score
            best_points = pts_final.copy()

    # Final polish on best
    if time.time() < TIME_LIMIT - 5:
        best_points = _polish(best_points, max_iter=300)
        final_score = _compute_covering_radius(best_points)
        if final_score < best_score:
            best_score = final_score

    return _normalize(best_points)
# EVOLVE-BLOCK-END
