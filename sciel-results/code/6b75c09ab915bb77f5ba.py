"""Fresh design: oriented-circle non-overlap + differential evolution + exact enclosing hexagon."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12
INR = np.sqrt(3.0) / 2.0          # inradius of a unit hexagon
MIN_DIST = 2.0 * INR              # = sqrt(3); center distance guaranteeing no overlap (shared orientation)
T0 = time.time()
DEADLINE = 150.0                  # wall-clock kill margin (evaluator kills at 150 s)


def _enclosing_side(pts, theta):
    """Minimal side length of a regular hexagon (centered at origin, orientation theta)
    enclosing all points in pts. Returns (side, center_offset)."""
    pts = np.asarray(pts, dtype=float)
    # 3 outward normal directions of the hexagon (separated by 60 deg)
    d = np.stack([
        np.array([np.cos(theta), np.sin(theta)]),
        np.array([np.cos(theta + np.pi / 3.0), np.sin(theta + np.pi / 3.0)]),
        np.array([np.cos(theta + 2.0 * np.pi / 3.0), np.sin(theta + 2.0 * np.pi / 3.0)]),
    ], axis=1)  # shape (3, 2)
    proj = pts @ d.T  # (n, 3)
    mx = proj.max(axis=0)
    mn = proj.min(axis=0)
    side = float((mx - mn).max())
    # offset so the hexagon is centered on the point cloud
    center = -0.5 * (mx + mn) @ d  # (2,)
    return side, center


def _build_solution(params, shared_angle):
    """params: (x0,y0,...,x11,y11). Returns (side, inner, center, angle_deg)."""
    pts = params.reshape(N, 2)
    side, center = _enclosing_side(pts, shared_angle)
    inner = [[float(pts[i, 0]), float(pts[i, 1]), float(np.degrees(shared_angle))] for i in range(N)]
    return side, inner, [float(center[0]), float(center[1])], float(np.degrees(shared_angle))


def _objective(params, shared_angle):
    """Objective: enclosing side length + heavy penalty for any overlap."""
    pts = params.reshape(N, 2)
    side, _ = _enclosing_side(pts, shared_angle)
    if N > 1:
        d = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((d ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        viol = np.maximum(0.0, MIN_DIST - dist).max()
        if viol > 0:
            return side + 1e6 * viol
    return side


def construct():
    # ---- seed initial population: a 3x4 grid of inradius circles (known disjoint) ----
    xs = np.array([0.0, MIN_DIST, 2.0 * MIN_DIST])
    ys = np.array([0.0, MIN_DIST, 2.0 * MIN_DIST, 3.0 * MIN_DIST])
    grid = np.stack([np.repeat(xs, 4), np.tile(ys, 3)], axis=1)  # 12 points
    grid = grid - grid.mean(axis=0)
    best_side, best_inner, best_center, best_angle = _build_solution(grid, 0.0)

    if time.time() - T0 > DEADLINE - 20.0:
        return best_inner, best_center, best_side, 0.0

    bounds = [(-3.0, 3.0)] * (2 * N)
    # shared rotation: a hexagon's enclosing side is invariant under 60 deg, so 0..pi/3 suffices
    shared_angle = 0.0

    # Try several shared angles to let the cloud adapt its shape.
    for sa in [0.0, np.pi / 12.0, np.pi / 6.0, np.pi / 9.0]:
        if time.time() - T0 > DEADLINE - 25.0:
            break
        try:
            res = differential_evolution(
                _objective,
                bounds,
                args=(sa,),
                seed=42,
                maxiter=120,
                popsize=10,
                tol=1e-4,
                polish=False,
                init="latinhypercube",
            )
        except Exception:
            continue
        if res.fun < best_side and res.fun < 1e5:  # ignore heavily-penalized (overlapping) results
            best_side, best_inner, best_center, best_angle = _build_solution(res.x, sa)

    return best_inner, best_center, best_side, best_angle
