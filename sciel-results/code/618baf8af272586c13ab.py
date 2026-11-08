"""Heilbronn triangles: maximize the smallest triangle area for 11 points in an equilateral triangle."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

# Precompute all C(11,3)=165 triangle index triples
_combos = list(combinations(range(N), 3))
_ci = np.array([c[0] for c in _combos])
_cj = np.array([c[1] for c in _combos])
_ck = np.array([c[2] for c in _combos])


def _min_area(pts):
    """Minimum area of any triangle formed by 3 of the 11 points."""
    pi = pts[_ci]
    pj = pts[_cj]
    pk = pts[_ck]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return float(np.min(areas))


def _inside_triangle(pts):
    """Check all points are inside (or on boundary of) the equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    # y >= 0
    if np.any(y < -1e-9):
        return False
    # y <= _H * x  (left edge)
    if np.any(y > _H * x + 1e-9):
        return False
    # y <= _H * (1 - x)  (right edge)
    if np.any(y > _H * (1.0 - x) + 1e-9):
        return False
    # x >= 0 and x <= 1
    if np.any(x < -1e-9) or np.any(x > 1.0 + 1e-9):
        return False
    return True


def _project(pts):
    """Project points onto the triangle (clamp to feasible region)."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    # x in [0, 1]
    x = np.clip(x, 0.0, 1.0)
    # y >= 0
    y = np.maximum(y, 0.0)
    # y <= min(_H*x, _H*(1-x))
    upper = np.minimum(_H * x, _H * (1.0 - x))
    y = np.minimum(y, upper)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _neg_min_area(pts):
    """Objective for minimization: negative of minimum triangle area."""
    pts = _project(pts)
    return -_min_area(pts)


def _initial_config():
    """
    Construct a well-known good configuration for 11 points in an equilateral triangle.
    Uses a grid-like arrangement optimized for the Heilbronn problem.
    """
    # A known good configuration: 3-4-3-1 or similar layered arrangement
    # Let's use a structured grid within the triangle
    pts = []
    # Layer 0 (bottom): 3 points along the base
    for i in range(3):
        pts.append([0.15 + 0.35 * i, 0.05])
    # Layer 1: 4 points
    for i in range(4):
        pts.append([0.1 + 0.25 * i, 0.22])
    # Layer 2: 3 points
    for i in range(3):
        pts.append([0.2 + 0.3 * i, 0.42])
    # Layer 3 (top): 1 point
    pts.append([0.5, 0.55])

    pts = np.array(pts)
    pts = _project(pts)
    return pts


def _initial_config2():
    """Alternative initial configuration using barycentric grid."""
    pts = []
    # Use a 4x4 grid in barycentric coords, select 11 best
    # Barycentric: (a, b, c) with a+b+c=1, a,b,c >= 0
    # Point = a*(0,0) + b*(1,0) + c*(0.5, _H)
    # Grid values
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    candidates = []
    for a in vals:
        for b in vals:
            c = 1.0 - a - b
            if c >= -1e-9:
                c = max(0.0, c)
                x = b + 0.5 * c
                y = _H * c
                candidates.append([x, y])
    candidates = np.array(candidates)

    # Greedy: pick 11 points maximizing min distance
    selected = [candidates[0]]  # start with a corner-ish point
    # Actually, let's pick from interior
    # Sort by distance from center
    center = np.array([0.5, _H / 3.0])
    dists = np.linalg.norm(candidates - center, axis=1)
    order = np.argsort(dists)
    # Take points spread out
    selected = []
    for idx in order:
        if len(selected) >= N:
            break
        # Check min distance to selected
        if selected:
            min_d = min(np.linalg.norm(candidates[idx] - s) for s in selected)
        else:
            min_d = 1.0
        if min_d > 0.08:
            selected.append(candidates[idx])
    # If we don't have 11, fill in
    while len(selected) < N:
        for idx in order:
            if candidates[idx] not in selected and all(
                np.linalg.norm(candidates[idx] - s) > 0.03 for s in selected
            ):
                selected.append(candidates[idx])
                break
        else:
            selected.append(candidates[order[len(selected)]])

    return np.array(selected[:N])


def _local_optimize(pts, max_time=60.0):
    """Run local optimization starting from pts."""
    best_pts = pts.copy()
    best_score = _min_area(pts)
    t0 = time.time()

    # Use L-BFGS-B on flattened coordinates
    x0 = pts.flatten()
    bounds = [(0.0, 1.0)] * (2 * N)

    def obj(x):
        p = x.reshape(N, 2)
        return _neg_min_area(p)

    try:
        res = minimize(
            obj, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'maxfun': 2000}
        )
        if time.time() - t0 < max_time:
            p = res.x.reshape(N, 2)
            p = _project(p)
            s = _min_area(p)
            if s > best_score:
                best_score = s
                best_pts = p.copy()
    except Exception:
        pass

    # Try a few random perturbations and re-optimize
    rng = np.random.RandomState(42)
    for trial in range(20):
        if time.time() - t0 > max_time - 10:
            break
        perturbed = best_pts + rng.randn(N, 2) * 0.02
        perturbed = _project(perturbed)
        x0 = perturbed.flatten()
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'maxfun': 1000}
            )
            if time.time() - t0 > max_time - 5:
                break
            p = res.x.reshape(N, 2)
            p = _project(p)
            s = _min_area(p)
            if s > best_score:
                best_score = s
                best_pts = p.copy()
        except Exception:
            pass

    return best_pts, best_score


def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s kill

    # Start with structured initial configurations
    candidates = []
    for init_fn in [_initial_config, _initial_config2]:
        try:
            pts = init_fn()
            if _inside_triangle(pts):
                candidates.append(pts)
        except Exception:
            pass

    # Also try a simple random start
    rng = np.random.RandomState(123)
    for _ in range(5):
        pts = rng.rand(N, 2)
        pts[:, 0] *= 1.0
        pts[:, 1] *= _H
        pts = _project(pts)
        if _inside_triangle(pts):
            candidates.append(pts)

    if not candidates:
        # Fallback: simple grid
        pts = np.array([[0.5 * (i % 3), _H * 0.1 * (i // 3)] for i in range(N)])
        return _project(pts)

    # Evaluate and pick best initial
    best_pts = candidates[0]
    best_score = _min_area(best_pts)
    for pts in candidates[1:]:
        s = _min_area(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Local optimization
    if time.time() < deadline - 60:
        best_pts, best_score = _local_optimize(best_pts, max_time=min(60, deadline - time.time() - 10))

    # Final validation
    best_pts = _project(best_pts)
    if not _inside_triangle(best_pts):
        # Last resort fallback
        best_pts = _project(_initial_config())

    return best_pts
