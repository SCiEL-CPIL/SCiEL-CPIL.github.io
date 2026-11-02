"""Covering the sphere with 130 equal caps: optimize convex-hull facet circumradii."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130
MAX_TIME = 140.0  # wall-clock budget with safety margin (limit is 150s)


def _initial_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return pts / n


def _facet_circumradius(p1, p2, p3):
    """Circumradius of the spherical triangle (p1,p2,p3) on the unit sphere,
    i.e. angular distance from the triangle's circumcenter to its vertices.
    If the triangle contains the origin (points in a closed hemisphere issue),
    the circumradius is pi/2."""
    a = np.dot(p1, p2)
    b = np.dot(p2, p3)
    c = np.dot(p3, p1)
    # circumcenter direction: cross product combination
    # circumcenter = (a*(p2 x p3) + b*(p3 x p1) + c*(p1 x p2)) / (a*b*c + 2*a*b*c ...)
    # Standard formula for circumcenter of spherical triangle:
    # Using the formula: circumcenter is proportional to
    # (sin(A) * p1 + sin(B) * p2 + sin(C) * p3) but let's use vector formula.
    # Vector circumcenter: solve for x = alpha*p1 + beta*p2 + gamma*p3
    # such that |x - p1| = |x - p2| = |x - p3| on the sphere.
    # Equivalently, x is equidistant from p1, p2, p3 in Euclidean sense
    # (since on unit sphere, angular distance = arccos(dot)).
    # So we want x . p1 = x . p2 = x . p3 = cos(R).
    # x = a1*p1 + a2*p2 + a3*p3, with x.p1 = x.p2 = x.p3.
    # This gives: a1 + a2*dot(p1,p2) + a3*dot(p1,p3) = a1*dot(p1,p2) + a2 + a3*dot(p2,p3)
    #            a1*dot(p1,p3) + a2*dot(p2,p3) + a3 = a1*dot(p1,p2) + a2*dot(p2,p1) + a3*dot(p3,p2)
    # Let's set up: (I - P) a = 0 where P is the 3x3 dot product matrix, but with constraint.
    # Actually, x must be in span{p1,p2,p3} and equidistant.
    # x = M * w where M = [p1 p2 p3], and x.p1 = x.p2 = x.p3 means M^T M w = M^T x = M^T M w = (M^T M w)
    # x.p_i = w^T M^T p_i = w^T (M^T M)_i. So we need (M^T M w)_1 = (M^T M w)_2 = (M^T M w)_3.
    # Let D = M^T M. We need D w = lambda * ones for some lambda. So w = lambda * D^{-1} ones.
    # Then x = M D^{-1} ones (up to scale).
    M = np.column_stack([p1, p2, p3])
    D = M.T @ M
    try:
        w = np.linalg.solve(D, np.ones(3))
    except np.linalg.LinAlgError:
        return np.pi / 2.0
    x = M @ w
    nx = np.linalg.norm(x)
    if nx < 1e-12:
        return np.pi / 2.0
    x = x / nx
    # Check that the circumcenter is on the "correct" side (same hemisphere as the triangle)
    # The circumradius is arccos(x . p1)
    cosR = x[0] * p1[0] + x[1] * p1[1] + x[2] * p1[2]
    cosR = np.clip(cosR, -1.0, 1.0)
    R = np.arccos(cosR)
    # If the triangle is too large (contains origin), the circumcenter may be on the wrong side.
    # Check: the facet normal should point away from origin. For a valid covering,
    # the circumcenter must be on the same side as the facet normal.
    # If R > pi/2, something is wrong.
    if R > np.pi / 2.0:
        # Try the other side
        x = -x
        cosR = x[0] * p1[0] + x[1] * p1[1] + x[2] * p1[2]
        cosR = np.clip(cosR, -1.0, 1.0)
        R = np.arccos(cosR)
        if R > np.pi / 2.0:
            return np.pi / 2.0
    return R


def _compute_circumradii(pts):
    """Compute circumradii for all facets of the convex hull."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    for k, tri in enumerate(simplices):
        p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        radii[k] = _facet_circumradius(p1, p2, p3)
    return radii, simplices


def _covering_radius(pts):
    radii, _ = _compute_circumradii(pts)
    if radii is None:
        return np.pi  # invalid
    return np.max(radii)


def _smooth_max(values, temperature=0.01):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(values)
    if m < -1e6:
        return m
    return m + np.log(np.sum(np.exp((values - m) / temperature))) * temperature


def _objective(pts, temperature=0.01):
    """Smooth objective: soft-max of facet circumradii."""
    radii, _ = _compute_circumradii(pts)
    if radii is None:
        return np.pi
    return _smooth_max(radii, temperature)


def _project_to_sphere(pts):
    return _normalize(pts)


def _gradient_descent(pts, max_iters=2000, lr_init=0.01, deadline=None):
    """Gradient descent on the sphere to minimize the smooth max of facet circumradii."""
    best_pts = pts.copy()
    best_obj = _covering_radius(pts)
    temperature = 0.02
    lr = lr_init

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        radii, simplices = _compute_circumradii(pts)
        if radii is None:
            break

        # Compute gradient of smooth max w.r.t. each point
        # d/dx_i [softmax(radii)] = sum_k softmax_weight_k * d(radius_k)/dx_i
        m = np.max(radii)
        exp_vals = np.exp((radii - m) / temperature)
        weights = exp_vals / np.sum(exp_vals)

        grad = np.zeros_like(pts)
        for k, tri in enumerate(simplices):
            if weights[k] < 1e-8:
                continue
            p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            # Gradient of circumradius w.r.t. each vertex
            # R = arccos(x . p1) where x is the circumcenter
            # dR/dp1 = -x / (sqrt(1 - (x.p1)^2)) * (d(x.p1)/dp1)
            # This is complex; use finite differences for simplicity? No, too slow.
            # Alternative: use the fact that for a spherical triangle,
            # the gradient of the circumradius w.r.t. a vertex is proportional
            # to the direction from the vertex to the circumcenter, projected
            # onto the tangent plane at the vertex.
            # Let's use a simpler approach: numerical gradient for the top few facets.
            pass

        # Since analytical gradient is complex, use a different strategy:
        # For each facet with high weight, push the vertices slightly toward
        # the circumcenter to reduce the circumradius.
        for k, tri in enumerate(simplices):
            if weights[k] < 1e-6:
                continue
            p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            # Compute circumcenter
            M = np.column_stack([p1, p2, p3])
            D = M.T @ M
            try:
                w = np.linalg.solve(D, np.ones(3))
            except np.linalg.LinAlgError:
                continue
            x = M @ w
            nx = np.linalg.norm(x)
            if nx < 1e-12:
                continue
            x = x / nx
            # For each vertex, move it slightly toward x (the circumcenter)
            # to reduce the angular distance to x.
            for j, idx in enumerate(tri):
                p = pts[idx]
                # Direction from p toward x, projected onto tangent plane at p
                dir_vec = x - np.dot(x, p) * p
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 1e-12:
                    dir_vec = dir_vec / dir_norm
                    # Small step in this direction
                    step = lr * weights[k] * 0.1
                    pts[idx] = pts[idx] + step * dir_vec
                    # Re-normalize
                    pts[idx] = pts[idx] / np.linalg.norm(pts[idx])

        # Re-normalize all points
        pts = _project_to_sphere(pts)

        # Evaluate
        obj = _covering_radius(pts)
        if obj < best_obj - 1e-10:
            best_obj = obj
            best_pts = pts.copy()

        # Anneal temperature
        temperature *= 0.999
        lr *= 0.999

    return best_pts, best_obj


def construct():
    t0 = time.time()
    deadline = t0 + MAX_TIME

    # Start with Fibonacci spiral
    pts = _initial_spiral(N)
    best_pts = pts.copy()
    best_obj = _covering_radius(pts)

    # Multiple restarts with perturbation
    rng = np.random.default_rng(42)
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        # Perturb the current best
        if restart == 0:
            cur = best_pts.copy()
        else:
            cur = best_pts + rng.normal(0, 0.05, size=(N, 3))
            cur = _project_to_sphere(cur)

        # Run gradient descent
        pts_opt, obj_opt = _gradient_descent(cur, max_iters=500, lr_init=0.005, deadline=deadline)

        if obj_opt < best_obj:
            best_obj = obj_opt
            best_pts = pts_opt

        # Check time
        if time.time() > deadline - 15.0:
            break

    # Final polish: a few more iterations with smaller learning rate
    if time.time() < deadline - 5.0:
        pts_final, obj_final = _gradient_descent(best_pts, max_iters=200, lr_init=0.001, deadline=deadline)
        if obj_final < best_obj:
            best_pts = pts_final
            best_obj = obj_final

    return _project_to_sphere(best_pts)
