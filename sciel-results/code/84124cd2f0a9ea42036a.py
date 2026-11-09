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


def _build_sofa_grid():
    """Return (xs, ys, X, Y) for the sofa grid (450 x 300)."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return xs, ys, X, Y


def _eval_path(path, X, Y):
    """path: (n,3) array of (x,y,a_deg). Returns number of sofa grid points
    inside EVERY placement (including evaluator-added endpoints)."""
    # Add evaluator endpoints
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    # Densify: between consecutive rows, m = max(1, ceil(|a2-a1|)) equal steps in (x,y,a)
    dense = [p[0]]
    for i in range(len(p) - 1):
        a1 = p[i, 2]
        a2 = p[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        # m equal steps -> m+1 points including endpoints; add the interior m-1? 
        # "adds linear interpolations ... in m equal steps" -> the segment is split into m
        # equal sub-steps, giving m-1 interior points (endpoints already present).
        if m > 1:
            ts = np.linspace(0.0, 1.0, m + 1)
            for t in ts[1:-1]:
                row = p[i] * (1.0 - t) + p[i + 1] * t
                dense.append(row)
        dense.append(p[i + 1])
    dense = np.array(dense)

    inside = np.ones(X.shape, dtype=bool)
    for (cx, cy, a_deg) in dense:
        a = np.deg2rad(a_deg)
        c, s = np.cos(a), np.sin(a)
        # R(-a) applied to (w - (cx,cy)):
        # R(-a) = [[c, s], [-s, c]]
        dx = X - cx
        dy = Y - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        mask = _base_inside(u, v)
        inside &= mask
        if not inside.any():
            break
    return int(inside.sum())


def _score(path, X, Y):
    return _eval_path(path, X, Y) * 0.0001


def _initial_path():
    # Start from the seed path (score 1.0 baseline) as a good starting point.
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
        # sort
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if abs(vals[0] - vals[-1]) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        # reflect
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            # expand
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
            # contract
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                # shrink toward best
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
        best_i = int(np.argmin(vals))
    return simplex[best_i], vals[best_i]


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n,3) array of corridor inner-corner placements (x,y,a_deg).
    The evaluator prepends (0,0,0) and appends (-2,0,90), densifies, and computes
    the sofa as the intersection of all placements on a 450x300 grid. Score is the
    count of grid points times 0.0001. We maximize this via Nelder-Mead on the
    (x,y) coordinates of a fixed-angle path.
    """
    xs, ys, X, Y = _build_sofa_grid()
    best_path = _initial_path()
    best_score = _score(best_path, X, Y)

    # Use a fixed set of angles (interior) and optimize the (x,y) at each.
    # Angles must be strictly between 0 and 90 to be valid interior placements.
    angles = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    n_ang = len(angles)

    # Initial (x,y) from the seed path, resampled to these angles.
    seed = best_path
    xy0 = np.zeros(2 * n_ang)
    for k, a in enumerate(angles):
        # find nearest seed angle
        idx = int(np.argmin(np.abs(seed[:, 2] - a)))
        xy0[2 * k] = seed[idx, 0]
        xy0[2 * k + 1] = seed[idx, 1]

    def f(flat):
        path = np.empty((n_ang, 3))
        path[:, 0] = flat[0::2]
        path[:, 1] = flat[1::2]
        path[:, 2] = angles
        return -_score(path, X, Y)  # minimize negative score

    # Nelder-Mead
    xopt, fopt = _nelder_mead(f, xy0, max_iter=2500, step=0.05)
    opt_path = np.empty((n_ang, 3))
    opt_path[:, 0] = xopt[0::2]
    opt_path[:, 1] = xopt[1::2]
    opt_path[:, 2] = angles
    opt_score = _score(opt_path, X, Y)
    if opt_score > best_score:
        best_path = opt_path
        best_score = opt_score

    # Local refinement: try a couple of different angle grids and restarts.
    if _elapsed() < _DEADLINE - 60:
        alt_angles = [
            np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]),
            np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]),
        ]
        for angles2 in alt_angles:
            if _elapsed() > _DEADLINE - 40:
                break
            n2 = len(angles2)
            xy2 = np.zeros(2 * n2)
            for k, a in enumerate(angles2):
                idx = int(np.argmin(np.abs(best_path[:, 2] - a)))
                xy2[2 * k] = best_path[idx, 0]
                xy2[2 * k + 1] = best_path[idx, 1]

            def f2(flat):
                path = np.empty((n2, 3))
                path[:, 0] = flat[0::2]
                path[:, 1] = flat[1::2]
                path[:, 2] = angles2
                return -_score(path, X, Y)

            xopt2, fopt2 = _nelder_mead(f2, xy2, max_iter=1500, step=0.04)
            p2 = np.empty((n2, 3))
            p2[:, 0] = xopt2[0::2]
            p2[:, 1] = xopt2[1::2]
            p2[:, 2] = angles2
            s2 = _score(p2, X, Y)
            if s2 > best_score:
                best_path = p2
                best_score = s2

    # Final sanity: ensure finite, shape (n,3), n>=1.
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = _initial_path()
    if not np.all(np.isfinite(best_path)):
        best_path = _initial_path()
    return best_path


# EVOLVE-BLOCK-END
