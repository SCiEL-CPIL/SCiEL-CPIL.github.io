# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 620.0  # seconds, safety margin under 630s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _base_inside(u, v):
    """Boolean: is (u,v) in the base L-shaped corridor?
    Base = {0<=u<=1, -3<=v<=1} U {-3<=u<=1, 0<=v<=1}."""
    horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return horiz | vert


def _build_sofa_grid(step=0.01):
    """Return (xs, ys, X, Y) for the sofa grid."""
    xs = np.arange(-3.0, 1.5, step)
    ys = np.arange(-1.5, 1.5, step)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return xs, ys, X, Y


def _densify_path(path):
    """Add evaluator endpoints and densify. Returns dense (m,3) array."""
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    dense = [p[0]]
    for i in range(len(p) - 1):
        a1 = p[i, 2]
        a2 = p[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        if m > 1:
            ts = np.linspace(0.0, 1.0, m + 1)
            for t in ts[1:-1]:
                row = p[i] * (1.0 - t) + p[i + 1] * t
                dense.append(row)
        dense.append(p[i + 1])
    return np.array(dense)


def _eval_path_fast(path, X, Y):
    """Fast evaluation: check all points against all placements using full masks."""
    dense = _densify_path(path)
    
    Xf = X.ravel()
    Yf = Y.ravel()
    npts = Xf.size
    
    # Start with all points active
    inside = np.ones(npts, dtype=bool)
    
    for (cx, cy, a_deg) in dense:
        if not np.any(inside):
            break
        a = np.deg2rad(a_deg)
        c, s = np.cos(a), np.sin(a)
        dx = Xf - cx
        dy = Yf - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        mask = _base_inside(u, v)
        inside &= mask
    
    return int(np.sum(inside))


def _score(path, X, Y):
    return _eval_path_fast(path, X, Y) * 0.0001


def _initial_path():
    return np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)


def _cubic_spline_path(ctrl_pts, n_samples=50):
    """
    Generate a smooth path from cubic spline control points.
    ctrl_pts: (4, 3) array of control points (x, y, a) for a cubic Bezier.
    Returns (n_samples, 3) array of path points.
    """
    # Cubic Bezier: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
    t = np.linspace(0.0, 1.0, n_samples)
    t2 = t ** 2
    t3 = t ** 3
    omt = 1.0 - t
    omt2 = omt ** 2
    omt3 = omt ** 3
    
    # We want to interpolate between start (0,0,0) and end (-2,0,90)
    # So we fix P0 = (0,0,0) and P3 = (-2,0,90), optimize P1 and P2
    P0 = np.array([0.0, 0.0, 0.0])
    P3 = np.array([-2.0, 0.0, 90.0])
    P1 = ctrl_pts[0:3]
    P2 = ctrl_pts[3:6]
    
    path = np.zeros((n_samples, 3))
    for i in range(3):
        path[:, i] = omt3 * P0[i] + 3 * omt2 * t * P1[i] + 3 * omt * t2 * P2[i] + t3 * P3[i]
    
    return path


def _nelder_mead(f, x0, max_iter=3000, step=0.05, tol=1e-6):
    """Simple Nelder-Mead. x0 is a flat array; f takes a flat array."""
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    best_i = int(np.argmin(vals))
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 30:
            break
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if abs(vals[0] - vals[-1]) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
        best_i = int(np.argmin(vals))
    return simplex[best_i], vals[best_i]


def _optimize_spline(X, Y, ctrl0, max_iter=1500, step=0.05, n_samples=50):
    """Optimize spline control points P1 and P2 (6 parameters)."""
    
    def f(ctrl):
        path = _cubic_spline_path(ctrl, n_samples)
        return -_score(path, X, Y)
    
    xopt, fopt = _nelder_mead(f, ctrl0, max_iter=max_iter, step=step)
    opt_path = _cubic_spline_path(xopt, n_samples)
    return opt_path


def _optimize_discrete_path(X, Y, angles, xy0, max_iter=2000, step=0.05):
    """Optimize (x,y) for given fixed angles."""
    n_ang = len(angles)
    
    def f(flat):
        path = np.empty((n_ang, 3))
        path[:, 0] = flat[0::2]
        path[:, 1] = flat[1::2]
        path[:, 2] = angles
        return -_score(path, X, Y)
    
    xopt, fopt = _nelder_mead(f, xy0, max_iter=max_iter, step=step)
    opt_path = np.empty((n_ang, 3))
    opt_path[:, 0] = xopt[0::2]
    opt_path[:, 1] = xopt[1::2]
    opt_path[:, 2] = angles
    return opt_path


def _final_polish(path, X, Y, max_iter=1000, step=0.01):
    """High-precision local refinement of the best path."""
    n = len(path)
    angles = path[:, 2].copy()
    xy0 = np.zeros(2 * n)
    for k in range(n):
        xy0[2 * k] = path[k, 0]
        xy0[2 * k + 1] = path[k, 1]
    
    def f(flat):
        p = np.empty((n, 3))
        p[:, 0] = flat[0::2]
        p[:, 1] = flat[1::2]
        p[:, 2] = angles
        return -_score(p, X, Y)
    
    xopt, fopt = _nelder_mead(f, xy0, max_iter=max_iter, step=step, tol=1e-8)
    opt_path = np.empty((n, 3))
    opt_path[:, 0] = xopt[0::2]
    opt_path[:, 1] = xopt[1::2]
    opt_path[:, 2] = angles
    return opt_path


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n,3) array of corridor inner-corner placements (x,y,a_deg).
    """
    # Use full precision grid
    xs, ys, X, Y = _build_sofa_grid(0.01)
    
    best_path = _initial_path()
    best_score = _score(best_path, X, Y)
    
    # Strategy 1: Spline-based smooth path optimization
    # Initial control points: P1 and P2 for cubic Bezier
    # P0 = (0,0,0), P3 = (-2,0,90)
    # Try different initial P1, P2
    spline_inits = [
        np.array([-0.5, 0.3, 30.0, -1.5, 0.3, 60.0]),
        np.array([-0.3, 0.2, 20.0, -1.7, 0.2, 70.0]),
        np.array([-0.7, 0.4, 40.0, -1.3, 0.4, 50.0]),
        np.array([-1.0, 0.5, 45.0, -1.0, 0.5, 45.0]),
    ]
    
    for ctrl0 in spline_inits:
        if _elapsed() > _DEADLINE - 120:
            break
        
        # Coarse-to-fine: start with fewer samples
        for n_samp in [20, 30, 50]:
            if _elapsed() > _DEADLINE - 60:
                break
            opt_path = _optimize_spline(X, Y, ctrl0, max_iter=800, step=0.06, n_samples=n_samp)
            opt_score = _score(opt_path, X, Y)
            if opt_score > best_score:
                best_path = opt_path
                best_score = opt_score
                # Refine with more samples
                ctrl0 = ctrl0  # Keep same init for next refinement
    
    # Strategy 2: Discrete path optimization with multiple angle configs
    angle_configs = [
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
        np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]),
        np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]),
        np.array([8.0, 18.0, 28.0, 38.0, 48.0, 58.0, 68.0, 78.0, 88.0]),
        np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]),
        np.array([7.0, 17.0, 27.0, 37.0, 47.0, 57.0, 67.0, 77.0, 87.0]),
        np.array([10.0, 22.0, 34.0, 46.0, 58.0, 70.0, 82.0]),
        np.array([13.0, 26.0, 39.0, 52.0, 65.0, 78.0]),
    ]
    
    for angles in angle_configs:
        if _elapsed() > _DEADLINE - 60:
            break
        
        n_ang = len(angles)
        xy0 = np.zeros(2 * n_ang)
        for k, a in enumerate(angles):
            idx = int(np.argmin(np.abs(best_path[:, 2] - a)))
            xy0[2 * k] = best_path[idx, 0]
            xy0[2 * k + 1] = best_path[idx, 1]
        
        # Multi-scale: coarse then fine
        opt_path = _optimize_discrete_path(X, Y, angles, xy0, max_iter=800, step=0.06)
        opt_score = _score(opt_path, X, Y)
        
        # Fine refinement
        if opt_score > best_score and _elapsed() < _DEADLINE - 40:
            xy1 = np.zeros(2 * n_ang)
            for k in range(n_ang):
                xy1[2 * k] = opt_path[k, 0]
                xy1[2 * k + 1] = opt_path[k, 1]
            opt_path2 = _optimize_discrete_path(X, Y, angles, xy1, max_iter=800, step=0.02)
            opt_score2 = _score(opt_path2, X, Y)
            if opt_score2 > best_score:
                best_path = opt_path2
                best_score = opt_score2
        elif opt_score > best_score:
            best_path = opt_path
            best_score = opt_score
    
    # Final high-precision polish
    if _elapsed() < _DEADLINE - 30:
        polished = _final_polish(best_path, X, Y, max_iter=1000, step=0.01)
        polished_score = _score(polished, X, Y)
        if polished_score > best_score:
            best_path = polished
            best_score = polished_score
    
    # Additional random restarts if time permits
    if _elapsed() < _DEADLINE - 20:
        for _ in range(3):
            if _elapsed() > _DEADLINE - 10:
                break
            n_ang = len(best_path)
            angles = best_path[:, 2].copy()
            xy0 = np.zeros(2 * n_ang)
            for k in range(n_ang):
                xy0[2 * k] = best_path[k, 0]
                xy0[2 * k + 1] = best_path[k, 1]
            
            # Add small random perturbation
            xy0 += np.random.normal(0, 0.02, size=2 * n_ang)
            
            def f(flat):
                path = np.empty((n_ang, 3))
                path[:, 0] = flat[0::2]
                path[:, 1] = flat[1::2]
                path[:, 2] = angles
                return -_score(path, X, Y)
            
            xopt, fopt = _nelder_mead(f, xy0, max_iter=400, step=0.02)
            opt_path = np.empty((n_ang, 3))
            opt_path[:, 0] = xopt[0::2]
            opt_path[:, 1] = xopt[1::2]
            opt_path[:, 2] = angles
            opt_score = _score(opt_path, X, Y)
            if opt_score > best_score:
                best_path = opt_path
                best_score = opt_score
    
    # Final sanity: ensure finite, shape (n,3), n>=1.
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = _initial_path()
    if not np.all(np.isfinite(best_path)):
        best_path = _initial_path()
    return best_path


# EVOLVE-BLOCK-END
