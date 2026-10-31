"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13

# EVOLVE-BLOCK-START

def _all_triangles(p):
    """Return areas of all triangles formed by triples of points."""
    n = len(p)
    idx = list(itertools.combinations(range(n), 3))
    m = len(idx)
    areas = np.zeros(m)
    for t, (i, j, k) in enumerate(idx):
        a, b, c = p[i], p[j], p[k]
        areas[t] = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
    return areas

def _hull_area(p):
    """Compute convex hull area."""
    try:
        h = ConvexHull(p)
        return h.volume
    except Exception:
        return 0.0

def _score(p):
    """Score = min triangle area / hull area."""
    ha = _hull_area(p)
    if ha < 1e-14:
        return 0.0
    areas = _all_triangles(p)
    m = np.min(areas)
    if m < 1e-14:
        return 0.0
    return m / ha

def _normalize(p):
    """Normalize so hull area = 1."""
    ha = _hull_area(p)
    if ha < 1e-14:
        return p
    return p / np.sqrt(ha)

def _regular_polygon(n, r=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

def _radial_to_cartesian(params):
    """Convert (angles, radii) to Cartesian points.
    params: array of shape (2N,) = [theta_0, ..., theta_{N-1}, r_0, ..., r_{N-1}]
    """
    thetas = params[:N]
    radii = params[N:]
    # Sort by angle to ensure convex polygon vertex order
    order = np.argsort(thetas)
    thetas = thetas[order]
    radii = radii[order]
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.column_stack([x, y])

def _soft_min_objective(params, tau, deadline):
    """Soft-min surrogate for radial representation."""
    if time.time() > deadline:
        return 1e10
    p = _radial_to_cartesian(params)
    ha = _hull_area(p)
    if ha < 1e-14:
        return 1e10
    areas = _all_triangles(p)
    # Soft min: -tau * log(sum(exp(-areas/tau)))
    logsum = np.log(np.sum(np.exp(-areas / tau)))
    soft_min = -tau * logsum
    if soft_min < 1e-14:
        return 1e10
    return -(soft_min / ha)

def _hard_objective(params, deadline):
    """Hard objective for radial representation."""
    if time.time() > deadline:
        return 1e10
    p = _radial_to_cartesian(params)
    ha = _hull_area(p)
    if ha < 1e-14:
        return 1e10
    areas = _all_triangles(p)
    m = np.min(areas)
    if m < 1e-14:
        return 1e10
    return -(m / ha)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    rng = np.random.default_rng(12345)

    # Get initial score from regular polygon
    reg_poly = _regular_polygon(13)
    reg_poly = _normalize(reg_poly)
    best_score = _score(reg_poly)
    best_points = reg_poly.copy()

    # Define bounds for differential evolution in radial space
    # Angles: [0, 2*pi), Radii: [0.1, 2.0]
    angle_bounds = [(0, 2 * np.pi)] * N
    radius_bounds = [(0.1, 2.0)] * N
    bounds = angle_bounds + radius_bounds

    # Initial population for DE
    # Start with regular polygon angles and radii = 1
    theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r0 = np.ones(N)
    x0 = np.concatenate([theta0, r0])

    # Differential Evolution on radial parameters
    # Use soft-min objective for smoothness
    tau = 0.05
    try:
        de_result = differential_evolution(
            _soft_min_objective,
            bounds,
            args=(tau, deadline),
            x0=x0,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            seed=12345,
            polish=False,
            init='latinhypercube'
        )
        if de_result.fun < -best_score + 1e-8:
            new_pts = _radial_to_cartesian(de_result.x)
            new_pts = _normalize(new_pts)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_points = new_pts
    except Exception:
        pass

    # Refine with L-BFGS-B on radial parameters using hard objective
    if time.time() < deadline - 10:
        x0 = de_result.x if 'de_result' in locals() and de_result.success else x0
        try:
            lbfgs_result = minimize(
                _hard_objective,
                x0,
                args=(deadline,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            new_pts = _radial_to_cartesian(lbfgs_result.x)
            new_pts = _normalize(new_pts)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_points = new_pts
        except Exception:
            pass

    # Final refinement in Cartesian space with Nelder-Mead
    if time.time() < deadline - 5:
        x0 = best_points.flatten()
        try:
            nm_result = minimize(
                lambda x: _hard_objective(x, deadline) if x.size == 2*N else 1e10,
                x0,
                args=(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False}
            )
            # NM operates on Cartesian, so we need a different objective
            # Let's just use the direct Cartesian objective
            pass
        except Exception:
            pass

    # Better final refinement: direct Cartesian NM
    if time.time() < deadline - 5:
        def cart_obj(x):
            if time.time() > deadline:
                return 1e10
            p = x.reshape(N, 2)
            ha = _hull_area(p)
            if ha < 1e-14:
                return 1e10
            areas = _all_triangles(p)
            m = np.min(areas)
            if m < 1e-14:
                return 1e10
            return -(m / ha)

        x0 = best_points.flatten()
        try:
            nm_result = minimize(
                cart_obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False}
            )
            new_pts = _normalize(nm_result.x.reshape(N, 2))
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_points = new_pts
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
