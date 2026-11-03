"""Combined approach: multi-start scaffolding + smooth LSE covering-radius refinement + Riesz polish."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin below 150s

# EVOLVE-BLOCK-START

def _normalize(points):
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _covering_radius(points):
    """Exact covering radius (degrees) via convex hull facet circumcenters."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(0)
        n_samples = 60000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return float(np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))) * 180.0 / np.pi

    max_r = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    return float(np.degrees(max_r))

def _facet_data(points):
    """Return (radii, centers, simplices) for triangular facets."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([]), [], []

    radii = []
    centers = []
    simplices = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        radii.append(r)
        centers.append(v)
        simplices.append(simplex)

    return np.array(radii), centers, simplices

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Smooth surrogate: log-sum-exp of facet circumradii.
    Returns (value, gradient) with gradient in ambient R^3 (not projected).
    """
    points = _normalize(points)
    radii, centers, simplices = _facet_data(points)
    n = len(radii)
    if n == 0:
        return 0.0, np.zeros_like(points)

    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)

    # numerically stable LSE
    exps = np.exp(beta * (radii - m))
    lse = m + np.log(np.sum(exps)) / beta
    w = exps / np.sum(exps)

    grad = np.zeros_like(points)
    for i in range(n):
        wi = w[i]
        if wi < 1e-15:
            continue
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(max(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            # project to tangent plane at p
            g_tan = g - np.dot(g, p) * p
            grad[idx] += wi * g_tan

    return float(lse), grad

def _riesz_energy_grad(points, s=12.0):
    """
    Riesz s-energy (sum over pairs of angular distance^{-s}) and gradient (tangent-projected).
    Used as an auxiliary smoother to diversify local minima.
    """
    points = _normalize(points)
    n = len(points)
    dots = points @ points.T
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angles = np.arccos(dots)
    sins = np.sin(angles)
    sins = np.maximum(sins, 1e-15)

    ang_powers = angles ** (-s - 1)
    np.fill_diagonal(ang_powers, 0.0)
    np.fill_diagonal(sins, 1.0)
    np.fill_diagonal(dots, 1.0)

    weights = s * ang_powers / sins
    np.fill_diagonal(weights, 0.0)

    grad = weights @ points
    grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points

    iu = np.triu_indices(n, k=1)
    energy = float(np.sum(angles[iu] ** (-s)))
    return energy, grad_tan

def _lbfgs_smooth(points, deadline=None, beta=50.0, maxiter=300,
                  ftol=1e-14, gtol=1e-12, maxfun=500):
    """
    L-BFGS-B on smooth LSE surrogate.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    def obj_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        # project gradient to tangent space
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "gtol": gtol,
                "maxfun": maxfun,
            }
        )
        x_opt = res.x.reshape(-1, 3)
        x_opt = _normalize(x_opt)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    if deadline is not None and time.time() >= deadline:
        return best, best_cr

    return best, best_cr

def _lbfgs_riesz(points, deadline=None, s=12.0, maxiter=200,
                 ftol=1e-12, gtol=1e-10, maxfun=400):
    """
    L-BFGS-B on Riesz energy (auxiliary smoother).
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    def obj_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _riesz_energy_grad(x, s=s)
        # grad already tangent-projected
        return val, grad.flatten()

    x0 = points.flatten()
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "gtol": gtol,
                "maxfun": maxfun,
            }
        )
        x_opt = res.x.reshape(-1, 3)
        x_opt = _normalize(x_opt)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    if deadline is not None and time.time() >= deadline:
        return best, best_cr

    return best, best_cr

def _lloyd_iteration(points, max_iters=300, tol=1e-10, deadline=None):
    """
    Spherical Lloyd: move each point toward the mean of circumcenters of incident facets.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    for _ in range(max_iters):
        if deadline is not None and time.time() >= deadline:
            break

        try:
            hull = ConvexHull(points)
        except Exception:
            break

        new_points = points.copy()
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)

        for i in range(len(points)):
            if not point_to_facets[i]:
                continue
            centers = [hull.equations[fi] for fi in point_to_facets[i]]
            # equational form: [n, d] where n is normal, d is offset
            # For unit sphere, normal is the circumcenter direction
            # Actually, let's use the facet_data approach for consistency
            pass

        # Use facet_data for consistency
        radii, centers, simplices = _facet_data(points)
        point_centers = [[] for _ in range(len(points))]
        for i in range(len(radii)):
            for idx in simplices[i]:
                point_centers[idx].append(centers[i])

        new_points = points.copy()
        for i in range(len(points)):
            if point_centers[i]:
                c = np.mean(point_centers[i], axis=0)
                cn = np.linalg.norm(c)
                if cn > 1e-12:
                    new_points[i] = c / cn

        alpha = 0.5
        blended = (1 - alpha) * points + alpha * new_points
        blended = _normalize(blended)
        cr = _covering_radius(blended)
        if cr < best_cr:
            points = blended
            best = points.copy()
            best_cr = cr
        else:
            alpha = 0.2
            blended = (1 - alpha) * points + alpha * new_points
            blended = _normalize(blended)
            cr = _covering_radius(blended)
            if cr < best_cr:
                points = blended
                best = points.copy()
                best_cr = cr
            else:
                break

    return best, best_cr

def _worst_facet_shrinkage(points, max_iters=500, deadline=None):
    """
    Deterministic iterative method: identify the convex hull facet with the largest
    spherical circumradius and move its three vertices along the great circle toward
    the facet's circumcenter.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    for it in range(max_iters):
        if deadline is not None and time.time() >= deadline:
            break

        radii, centers, simplices = _facet_data(points)
        if len(radii) == 0:
            break

        worst_idx = np.argmax(radii)
        worst_r = radii[worst_idx]
        worst_center = centers[worst_idx]
        worst_simplex = simplices[worst_idx]

        if worst_r < 1e-10:
            break

        # Move the three vertices of the worst facet toward the circumcenter
        step_size = 0.3
        new_points = points.copy()
        for idx in worst_simplex:
            p = points[idx]
            c = worst_center
            cos_angle = np.clip(np.dot(p, c), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < 1e-10:
                continue
            t = step_size
            sin_angle = np.sin(angle)
            if sin_angle < 1e-10:
                continue
            new_p = (np.sin((1-t)*angle) * p + np.sin(t*angle) * c) / sin_angle
            new_points[idx] = new_p

        new_points = _normalize(new_points)
        new_cr = _covering_radius(new_points)

        if new_cr < best_cr - 1e-12:
            points = new_points
            best = points.copy()
            best_cr = new_cr
        else:
            step_size *= 0.5
            if step_size < 0.01:
                break

    return best, best_cr

def _fibonacci_points(n=N):
    """Generate points using golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed):
    """Generate random points on sphere."""
    np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _normalize(pts)

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    global START_TIME
    START_TIME = time.time()
    deadline = START_TIME + TIME_LIMIT

    # Generate diverse initial configurations
    starts = []

    # 1. Golden-angle spiral
    starts.append(_fibonacci_points())

    # 2. Random starts with different seeds
    for seed in [1, 2, 3, 4, 5, 7, 13, 42, 99, 123]:
        starts.append(_random_points(seed))

    # 3. Perturbed golden spiral
    np.random.seed(7)
    base = starts[0].copy()
    noise = np.random.randn(N, 3) * 0.1
    starts.append(_normalize(base + noise))

    # 4. More perturbation
    np.random.seed(13)
    noise = np.random.randn(N, 3) * 0.3
    starts.append(_normalize(base + noise))

    # Optimize from each start, keep the best
    best_pts = starts[0]
    best_val = _covering_radius(best_pts)

    # Allocate time per start
    n_starts = len(starts)
    elapsed = time.time() - START_TIME
    time_per_start = max(5.0, (TIME_LIMIT - elapsed - 20.0) / n_starts)

    for idx, start in enumerate(starts):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        max_time_for_this = min(time_per_start, remaining - 5.0)
        if max_time_for_this < 3.0:
            break

        try:
            # Phase 1: Worst facet shrinkage (fast, deterministic)
            pts, val = _worst_facet_shrinkage(start, max_iters=100,
                                               deadline=time.time() + max_time_for_this * 0.2)
            if val < best_val:
                best_val = val
                best_pts = pts

            # Phase 2: Lloyd iteration
            pts, val = _lloyd_iteration(pts, max_iters=50,
                                         deadline=time.time() + max_time_for_this * 0.4)
            if val < best_val:
                best_val = val
                best_pts = pts

            # Phase 3: L-BFGS refinement with multi-stage beta
            remaining_for_lbfgs = max(5.0, max_time_for_this * 0.6)
            sub_deadline = time.time() + remaining_for_lbfgs

            # Stage 1: low beta for global smoothing
            if time.time() < sub_deadline - 10:
                pts, val = _lbfgs_smooth(pts, deadline=sub_deadline - 10, beta=10.0, maxiter=100)
                if val < best_val:
                    best_val = val
                    best_pts = pts

            # Stage 2: medium beta
            if time.time() < sub_deadline - 5:
                pts, val = _lbfgs_smooth(pts, deadline=sub_deadline - 5, beta=50.0, maxiter=200)
                if val < best_val:
                    best_val = val
                    best_pts = pts

            # Stage 3: high beta for precise convergence
            if time.time() < sub_deadline:
                pts, val = _lbfgs_smooth(pts, deadline=sub_deadline, beta=200.0, maxiter=300)
                if val < best_val:
                    best_val = val
                    best_pts = pts

        except Exception:
            continue

    # Final polish with tighter tolerances if time allows
    remaining = deadline - time.time()
    if remaining > 15.0:
        try:
            # Riesz energy polish for diversity
            pts, val = _lbfgs_riesz(best_pts, deadline=time.time() + remaining * 0.3, s=12.0)
            if val < best_val:
                best_val = val
                best_pts = pts

            # More aggressive refinement
            pts, val = _worst_facet_shrinkage(best_pts, max_iters=300,
                                               deadline=time.time() + remaining * 0.5)
            if val < best_val:
                best_val = val
                best_pts = pts

            pts, val = _lloyd_iteration(pts, max_iters=200,
                                         deadline=time.time() + remaining * 0.7)
            if val < best_val:
                best_val = val
                best_pts = pts

            # Final high-precision LSE polish
            pts, val = _lbfgs_smooth(best_pts, deadline=time.time() + remaining * 0.9,
                                      beta=500.0, maxiter=500)
            if val < best_val:
                best_val = val
                best_pts = pts
        except Exception:
            pass

    # Final safety check
    if time.time() > deadline - 5.0:
        return _normalize(best_pts)

    return _normalize(best_pts)

# EVOLVE-BLOCK-END
