"""Combined approach: fast convex-hull based covering radius + force-based local search with deadline guard."""
N = 50
import time
import numpy as np
from scipy.spatial import ConvexHull


def _cover_radius(pts):
    """Compute covering radius (in radians) as max circumradius over hull facets."""
    if len(pts) < 3:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    max_angle = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]

        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)

        w_a = np.linalg.norm(cross_bc)
        w_b = np.linalg.norm(cross_ca)
        w_c = np.linalg.norm(cross_ab)

        total = w_a + w_b + w_c
        if total < 1e-15:
            continue

        circum = a * w_a + b * w_b + c * w_c
        cnorm = np.linalg.norm(circum)
        if cnorm < 1e-15:
            continue
        circum /= cnorm

        facet_normal = np.cross(b - a, c - a)
        fnorm = np.linalg.norm(facet_normal)
        if fnorm > 1e-15:
            facet_normal /= fnorm
            if np.dot(circum, facet_normal) < 0:
                circum = -circum

        dots = np.array([np.dot(circum, a), np.dot(circum, b), np.dot(circum, c)])
        min_dot = np.clip(min(dots), -1.0, 1.0)
        angle = np.arccos(min_dot)
        if angle > max_angle:
            max_angle = angle

    return max_angle


def _fibonacci_init():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _force_based_optimize(pts, max_iters=3000, deadline=None):
    """
    Force-based optimization: points are attracted toward worst facet circumcenters
    and repelled from each other when too close.
    """
    pts = pts.copy()
    best_radius = _cover_radius(pts)
    best_pts = pts.copy()

    step = 0.05
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        worst_radius = 0.0
        worst_circum = None

        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            cross_bc = np.cross(b, c)
            cross_ca = np.cross(c, a)
            cross_ab = np.cross(a, b)
            w_a = np.linalg.norm(cross_bc)
            w_b = np.linalg.norm(cross_ca)
            w_c = np.linalg.norm(cross_ab)
            total = w_a + w_b + w_c
            if total < 1e-15:
                continue
            circum = a * w_a + b * w_b + c * w_c
            cnorm = np.linalg.norm(circum)
            if cnorm < 1e-15:
                continue
            circum /= cnorm
            facet_normal = np.cross(b - a, c - a)
            fnorm = np.linalg.norm(facet_normal)
            if fnorm > 1e-15:
                facet_normal /= fnorm
                if np.dot(circum, facet_normal) < 0:
                    circum = -circum
            dots = np.array([np.dot(circum, a), np.dot(circum, b), np.dot(circum, c)])
            min_dot = np.clip(min(dots), -1.0, 1.0)
            angle = np.arccos(min_dot)
            if angle > worst_radius:
                worst_radius = angle
                worst_circum = circum

        if worst_circum is None:
            break

        forces = np.zeros_like(pts)

        dots_to_circum = pts @ worst_circum
        nearest_idx = np.argmin(dots_to_circum)
        forces[nearest_idx] += 0.5 * (worst_circum - pts[nearest_idx])

        # Vectorized repulsion
        dots_mat = pts @ pts.T
        np.fill_diagonal(dots_mat, -1.0)
        angles_mat = np.arccos(np.clip(dots_mat, -1.0, 1.0))
        rep_mask = angles_mat < 0.2
        rep_strengths = np.where(rep_mask, (0.2 - angles_mat) / 0.2, 0.0)

        for i in range(N):
            if not np.any(rep_mask[i]):
                continue
            js = np.where(rep_mask[i])[0]
            diffs = pts[i] - pts[js]
            norms = np.linalg.norm(diffs, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            diffs /= norms
            forces[i] += np.sum(rep_strengths[i, js, None] * diffs, axis=0)
            for j in js:
                forces[j] -= rep_strengths[i, j] * diffs[np.where(js == j)[0][0]]

        for i in range(N):
            fnorm = np.linalg.norm(forces[i])
            if fnorm > 1e-10:
                forces[i] /= fnorm

        new_pts = pts + step * forces
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)

        new_radius = _cover_radius(new_pts)
        if new_radius < best_radius - 1e-10:
            pts = new_pts
            best_radius = new_radius
            best_pts = pts.copy()
        else:
            step *= 0.85
            if step < 1e-6:
                break

    return best_pts, best_radius


def _hill_climbing(pts, max_iters=500, deadline=None):
    """Hill climbing: try moving each point in random tangent directions, keep improvements."""
    pts = pts.copy()
    best_radius = _cover_radius(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(42)
    step = 0.015

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            random_dir = rng.randn(3)
            tangent = random_dir - np.dot(random_dir, pts[i]) * pts[i]
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-10:
                continue
            tangent /= tnorm

            new_pts = pts.copy()
            new_pts[i] = pts[i] + step * tangent
            new_pts[i] /= np.linalg.norm(new_pts[i])

            new_radius = _cover_radius(new_pts)
            if new_radius < best_radius - 1e-10:
                pts = new_pts
                best_radius = new_radius
                best_pts = pts.copy()
                improved = True

        if not improved:
            step *= 0.8
            if step < 1e-7:
                break

    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    # Start with Fibonacci spiral
    pts = _fibonacci_init()

    # Phase 1: Force-based optimization
    pts, _ = _force_based_optimize(pts, max_iters=2000, deadline=deadline)

    # Phase 2: Hill climbing for fine-tuning
    if time.time() < deadline - 10.0:
        pts, _ = _hill_climbing(pts, max_iters=300, deadline=deadline)

    # Final check: ensure points are on unit sphere and non-degenerate
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)

    return pts
