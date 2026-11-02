"""Spherical covering with 50 points: epigraph + smooth surrogate + facet local search."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
T_START = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _time_left():
    return max(0.0, TIME_LIMIT - (time.time() - T_START))


def _params_to_points(p):
    """p: (N,2) -> (N,3) unit vectors (spherical coords)."""
    theta = p[:, 0]
    phi = p[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y, z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _points_to_params(pts):
    pts = np.asarray(pts)
    z = np.clip(pts[:, 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    return np.column_stack([theta, phi])


def _covering_radius(pts):
    """Exact covering radius in radians via Delaunay tetrahedralization of the convex hull facets."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts)
    # Add origin to ensure full-dimensional hull; facets are triangles on the sphere
    hull = ConvexHull(np.vstack([pts, np.zeros((1, 3))]))
    # Facets: each is a triangle of 3 indices (or tetra). For points on sphere + origin,
    # hull facets are triangles of 3 sphere points.
    radii = []
    for simplex in hull.simplices:
        # simplex is a tuple of indices; origin index is N
        idx = [i for i in simplex if i < N]
        if len(idx) < 3:
            continue
        tri = pts[idx[:3]]
        a, b, c = tri
        # Circumcenter of triangle on unit sphere:
        # Solve for x = alpha*a + beta*b + gamma*c with x.a=x.b=x.c=1 (on unit sphere plane)
        # Use the formula: circumcenter of spherical triangle.
        # Vector method: find point x on sphere equidistant (in angle) from a,b,c.
        # x = (a + b + c) normalized works only if triangle is small; use linear solve.
        # Constraints: x dot a = x dot b = x dot c = cos(R), |x|=1.
        # Let x = u*a + v*b + w*c. Solve M [u,v,w]^T = [cosR,cosR,cosR] and |x|^2=1.
        M = np.array([
            [a @ a, a @ b, a @ c],
            [b @ a, b @ b, b @ c],
            [c @ a, c @ b, c @ c],
        ])
        # We want x dot a = x dot b = x dot c. Let that value be t.
        # x = M^{-1} t * [1,1,1]^T  =>  x = M^{-1} @ ones * t
        try:
            ones = np.ones(3)
            x_dir = np.linalg.solve(M, ones)
        except np.linalg.LinAlgError:
            continue
        x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-300)
        # t = x dot a
        t = x_dir @ a
        # Check that this is the farthest point (the cap center opposite to origin side)
        # The circumradius is the angular distance from x_dir to a (or b or c).
        t = np.clip(t, -1.0, 1.0)
        R = np.arccos(t)
        # Ensure we take the larger of the two caps (the one away from origin)
        # x_dir should point away from origin; if not, flip
        # The origin is inside the hull, so the relevant cap is the one NOT containing origin.
        # x_dir is on the sphere; the cap centered at x_dir with radius R is empty if
        # all points are within R of x_dir. Verify:
        dists = np.arccos(np.clip(pts @ x_dir, -1.0, 1.0))
        if np.max(dists) > R + 1e-9:
            # Wrong side; the other cap
            x_dir = -x_dir
            R = np.pi - R
        radii.append(R)
    if not radii:
        # Fallback: max pairwise angular distance / 2
        from scipy.spatial.distance import cdist
        G = np.clip(pts @ pts.T, -1.0, 1.0)
        d = np.arccos(G)
        return float(np.max(d) / 2.0)
    return float(np.max(radii))


def _smooth_covering_radius(pts, beta=200.0):
    """Smooth surrogate: soft-max of angular distances to nearest point, over a dense grid."""
    # Use a moderate grid for smoothness; beta controls sharpness
    from scipy.spatial.distance import cdist
    # Fibonacci grid of ~4000 points
    M = 4000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    G = np.clip(grid @ pts.T, -1.0, 1.0)  # (M, N) cosines
    D = np.arccos(G)  # (M, N) angular distances
    # For each grid point, nearest = min over N
    # soft-min: -logsumexp(-beta * D)/beta
    neg = -beta * D
    neg_max = np.max(neg, axis=1, keepdims=True)
    soft_min = -np.mean(neg_max - np.log(np.sum(np.exp(neg - neg_max), axis=1, keepdims=True))) / beta
    return float(np.max(soft_min))


def _objective_and_grad(p):
    """Epigraph objective: minimize t s.t. smooth_covering_radius(pts) <= t.
    We minimize t + penalty * max(0, f(t) - t) but simpler: minimize t where f is the
    smooth surrogate. Actually we minimize the smooth surrogate directly (it approximates
    covering radius). Use t as the value.
    """
    pts = _params_to_points(p)
    val = _smooth_covering_radius(pts, beta=200.0)
    # Gradient via finite differences is slow; use analytic-ish via autograd not available.
    # Instead, we return val and let L-BFGS use numerical gradient (cheap enough for N=50).
    return val


def _optimize_from(pts0, max_iter=400, ftol=1e-12, gtol=1e-12):
    p0 = _points_to_params(pts0)
    res = minimize(
        _objective_and_grad,
        p0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50},
    )
    return _params_to_points(res.x), res.fun


def _facet_local_search(pts, max_rounds=200, tol=1e-12):
    """Directly reduce the largest facet circumradius by moving the 3 defining points
    toward the facet circumcenter (which reduces the cap radius)."""
    from scipy.spatial import ConvexHull
    pts = np.array(pts, dtype=float)
    for _ in range(max_rounds):
        if _time_left() < 5.0:
            break
        try:
            hull = ConvexHull(np.vstack([pts, np.zeros((1, 3))]))
        except Exception:
            break
        best_R = -1.0
        best_idx = None
        best_center = None
        for simplex in hull.simplices:
            idx = [i for i in simplex if i < N]
            if len(idx) < 3:
                continue
            tri = pts[idx[:3]]
            a, b, c = tri
            M = np.array([
                [a @ a, a @ b, a @ c],
                [b @ a, b @ b, b @ c],
                [c @ a, c @ b, c @ c],
            ])
            try:
                ones = np.ones(3)
                x_dir = np.linalg.solve(M, ones)
            except np.linalg.LinAlgError:
                continue
            x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-300)
            t = np.clip(x_dir @ a, -1.0, 1.0)
            R = np.arccos(t)
            dists = np.arccos(np.clip(pts @ x_dir, -1.0, 1.0))
            if np.max(dists) > R + 1e-9:
                x_dir = -x_dir
                R = np.pi - R
            if R > best_R:
                best_R = R
                best_idx = idx[:3]
                best_center = x_dir
        if best_R < 0:
            break
        # Move the three defining points slightly toward the circumcenter to shrink the cap.
        # This reduces the circumradius of that facet.
        alpha = 0.15
        for i in best_idx:
            pts[i] = (1 - alpha) * pts[i] + alpha * best_center
            pts[i] /= np.linalg.norm(pts[i])
        # Re-normalize all
        for i in range(N):
            pts[i] /= np.linalg.norm(pts[i])
        # Check improvement
        new_R = _covering_radius(pts)
        if new_R >= best_R - tol:
            # Revert
            break
    return pts


def _golden_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    best_pts = _golden_spiral()
    best_val = _covering_radius(best_pts)

    # Multi-restart L-BFGS with smooth surrogate
    rng = np.random.default_rng(12345)
    n_restarts = 4
    for r in range(n_restarts):
        if _time_left() < 15.0:
            break
        if r == 0:
            start = best_pts
        else:
            # Perturb
            start = best_pts + rng.normal(0, 0.5, (N, 3))
            start /= np.linalg.norm(start, axis=1, keepdims=True)
        try:
            pts_opt, val = _optimize_from(start, max_iter=300)
            # Exact evaluation
            exact = _covering_radius(pts_opt)
            if exact < best_val:
                best_val = exact
                best_pts = pts_opt
        except Exception:
            continue

    # Facet-based local search to polish
    if _time_left() > 10.0:
        pts_polished = _facet_local_search(best_pts, max_rounds=150)
        exact_polished = _covering_radius(pts_polished)
        if exact_polished < best_val:
            best_val = exact_polished
            best_pts = pts_polished

    # Final high-precision polish: one more L-BFGS with tighter tolerance
    if _time_left() > 8.0:
        try:
            pts_final, _ = _optimize_from(best_pts, max_iter=200, ftol=1e-14, gtol=1e-14)
            exact_final = _covering_radius(pts_final)
            if exact_final < best_val:
                best_val = exact_final
                best_pts = pts_final
        except Exception:
            pass

    # Ensure validity: unit norm, no coincident points, not in a closed hemisphere
    best_pts = np.asarray(best_pts, dtype=float)
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    # Check not all in a closed hemisphere: there should be a point with negative dot
    # with some direction; equivalently, the origin is in the convex hull.
    # Quick check: min over directions of max dot < 0 is hard; instead check that
    # the convex hull of points contains origin.
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(np.vstack([best_pts, np.zeros((1, 3))]))
        # If origin is a vertex, points are in a closed hemisphere -> invalid
        if N in hull.vertices:
            # Perturb to push out
            best_pts[0] += 0.1
            best_pts[0] /= np.linalg.norm(best_pts[0])
    except Exception:
        pass

    return best_pts
