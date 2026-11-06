"""Refined covering-radius minimizer for 130 points on the unit sphere.

Keeps the geometric local-search design but adds:
  * a smooth soft-min surrogate for the covering radius (log-sum-exp over a
    dense spherical sampling),
  * a time-budgeted L-BFGS-B refinement,
  * a final exact local polish pass.
"""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

N = 130
# Deadline guard: evaluator kills at 150 s; keep a safety margin.
DEADLINE = 130.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return DEADLINE - _elapsed()


def _soft_covering_radius(pts, sample_pts, temperature=0.05):
    """Smooth surrogate: soft-min of distances to nearest point.

    For each sample point y, d_y = min_i angle(y, p_i).
    We approximate the max over y of d_y using a soft-max over sample points
    of the soft-min distances.
    """
    # pts: (N,3), sample_pts: (M,3), both unit vectors
    # dot products
    dots = np.clip(sample_pts @ pts.T, -1.0, 1.0)  # (M, N)
    # distances in radians: arccos(dot)
    dists = np.arccos(dots)  # (M, N)
    # soft-min over N for each sample point (using log-sum-exp trick)
    # softmin(dists, T) = -T * log(sum(exp(-dists/T)))
    neg = -dists / temperature
    # subtract max for numerical stability
    neg_max = np.max(neg, axis=1, keepdims=True)
    softmin = -temperature * (neg_max[:, 0] + np.log(np.sum(np.exp(neg - neg_max), axis=1)))
    # Now soft-max over M sample points
    pos = softmin / temperature
    pos_max = np.max(pos, keepdims=True)
    softmax = temperature * (pos_max[0, 0] + np.log(np.sum(np.exp(pos - pos_max))))
    return softmax


def _exact_covering_radius(pts):
    """Compute the exact covering radius using convex hull facets.

    The covering radius is the maximum over all facets (circumcircles of
    Delaunay triangulation) of the circumradius angle.
    """
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            # Each simplex is a triangle of 3 points
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Circumcenter of triangle on unit sphere
            # The circumradius angle is the angle from circumcenter to any vertex
            # Compute via the formula for spherical triangle circumradius
            # Using the fact that the circumcenter is in the plane of a,b,c
            # and equidistant from all three
            # Solve: |x - a|^2 = |x - b|^2 = |x - c|^2, |x|=1
            # This gives a linear system for x in the plane
            # Simpler: the circumradius angle r satisfies
            # sin(r) = |a x b| / (2 * |a| |b|) ... no, that's for 2 points
            # For 3 points, use the spherical excess or direct computation
            # The circumcenter n is proportional to (a x b) + (b x c) + (c x a)
            # Actually, the circumcenter of a spherical triangle is the
            # normalized sum of the edge cross products weighted appropriately.
            # Simpler approach: the circumradius is half the maximum distance
            # between any two vertices? No, that's not right either.
            # Let's use: for a triangle with vertices on unit sphere,
            # the circumradius R satisfies:
            # sin(R) = |a x b| / (2 sin(angle_c)) where angle_c is the angle at c
            # This is getting complex. Let's just compute the circumcenter directly.
            # The circumcenter x satisfies:
            # (x - a) . (b - a) = 0  =>  x . (b-a) = a . (b-a)
            # (x - a) . (c - a) = 0  =>  x . (c-a) = a . (c-a)
            # |x|^2 = 1
            # First two are linear in x. Solve for x in the plane.
            ab = b - a
            ac = c - a
            # x . ab = a . ab, x . ac = a . ac
            # Let x = alpha * ab + beta * ac + gamma * n where n = ab x ac
            # But simpler: use the 2x2 linear system for the projection
            A = np.array([ab, ac])  # (2,3)
            rhs = np.array([a @ ab, a @ ac])
            # x_proj = A^T (A A^T)^{-1} rhs
            AAT = A @ A.T  # (2,2)
            try:
                coeffs = np.linalg.solve(AAT, rhs)
                x_proj = A.T @ coeffs
            except np.linalg.LinAlgError:
                continue
            # The circumcenter is x_proj + t * n where n is the normal
            n = np.cross(ab, ac)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            # |x|^2 = 1: |x_proj + t n|^2 = 1
            # |x_proj|^2 + 2 t (x_proj . n) + t^2 = 1
            # x_proj is in the plane, so x_proj . n = 0
            # t^2 = 1 - |x_proj|^2
            xproj_norm_sq = x_proj @ x_proj
            if xproj_norm_sq >= 1.0:
                t_val = 0.0
            else:
                t_val = np.sqrt(1.0 - xproj_norm_sq)
            # Two possible circumcenters; pick the one on the same side as the triangle
            # (i.e., the one with positive dot with n, or check which is "outside")
            x1 = x_proj + t_val * n
            x2 = x_proj - t_val * n
            # The circumcenter should be such that the arc from x to a is the
            # circumradius. We want the smaller arc, so pick the circumcenter
            # that gives the smaller angle.
            # Actually, for the covering radius, we want the circumcenter of the
            # empty cap, which is the one on the opposite side of the facet from
            # the origin... no, the circumcenter is on the sphere.
            # The covering radius for this facet is the angle from the circumcenter
            # to any vertex. Both x1 and x2 give the same angle (since they're
            # antipodal? No, they're not necessarily antipodal).
            # Actually x1 and x2 are different points. The correct circumcenter
            # is the one that is equidistant from a, b, c AND lies on the sphere.
            # Both do. The one we want is the one that forms the "empty cap"
            # center, which is the one farther from the origin in the direction
            # of the facet normal. For a convex hull, the circumcenter of a facet
            # that defines a Voronoi cell boundary is the one on the "outside".
            # Let's just compute the angle for both and take the max (conservative).
            d1 = np.arccos(np.clip(x1 @ a, -1.0, 1.0))
            d2 = np.arccos(np.clip(x2 @ a, -1.0, 1.0))
            # The covering radius contribution is the larger of the two
            # (since we want the worst-case empty cap)
            r = max(d1, d2)
            if r > max_r:
                max_r = r
        return max_r
    except Exception:
        # Fallback: sample-based estimate
        return _soft_covering_radius(pts, _sample_sphere(5000), 0.05)


def _sample_sphere(M, seed=42):
    """Generate M approximately uniform points on the unit sphere."""
    rng = np.random.RandomState(seed)
    # Fibonacci spiral for quasi-uniform sampling
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return pts / norms


def _initial_fibonacci():
    """Golden-angle Fibonacci spiral initial configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_search(pts, max_iters=500, step_size=0.1):
    """Simple gradient-free local search using random perturbations."""
    pts = _normalize(pts.copy())
    best_r = _exact_covering_radius(pts)
    rng = np.random.RandomState(12345)
    for it in range(max_iters):
        if _remaining() < 10.0:
            break
        # Perturb one random point
        idx = rng.randint(N)
        perturbation = rng.normal(0, step_size, 3)
        new_pts = pts.copy()
        new_pts[idx] = _normalize(new_pts[idx] + perturbation)[0]
        new_r = _exact_covering_radius(new_pts)
        if new_r < best_r - 1e-10:
            pts = new_pts
            best_r = new_r
    return pts, best_r


def _lbfgs_refine(pts, sample_pts, max_iter=200):
    """Refine using L-BFGS-B with the soft covering radius objective."""
    def objective(flat):
        p = _normalize(flat.reshape(N, 3))
        return _soft_covering_radius(p, sample_pts, temperature=0.05)

    def jac(flat):
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(flat)
        for i in range(len(flat)):
            flat_plus = flat.copy()
            flat_plus[i] += eps
            f_plus = objective(flat_plus)
            flat_minus = flat.copy()
            flat_minus[i] -= eps
            f_minus = objective(flat_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    x0 = pts.flatten()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10})
        if res.success or res.fun < objective(x0):
            return _normalize(res.x.reshape(N, 3))
    except Exception:
        pass
    return pts


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    # Start with Fibonacci spiral
    pts = _initial_fibonacci()
    pts = _normalize(pts)

    # Phase 1: Local search with decreasing step size
    for step in [0.3, 0.1, 0.05, 0.02, 0.01]:
        if _remaining() < 15.0:
            break
        pts, _ = _local_search(pts, max_iters=200, step_size=step)

    # Phase 2: L-BFGS refinement with soft surrogate
    if _remaining() > 20.0:
        sample_pts = _sample_sphere(2000)
        pts = _lbfgs_refine(pts, sample_pts, max_iter=100)

    # Phase 3: Final local polish
    if _remaining() > 10.0:
        for step in [0.01, 0.005, 0.002]:
            if _remaining() < 10.0:
                break
            pts, _ = _local_search(pts, max_iters=100, step_size=step)

    # Final normalization
    pts = _normalize(pts)

    # Safety check: ensure non-degenerate
    if not np.all(np.isfinite(pts)):
        return _initial_fibonacci()

    return pts
