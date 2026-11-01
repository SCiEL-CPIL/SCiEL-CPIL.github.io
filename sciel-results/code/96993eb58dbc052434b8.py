"""Heilbronn-type problem: place N points to maximize min triangle area / hull area."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
import time

N = 14
DEADLINE = 115  # seconds, safety margin before 120s kill


def _min_triangle_area(pts):
    """Return the minimum triangle area over all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Vectorized: compute cross products for all triples
    # For speed, do it in chunks or use a loop
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-15:
                        return 0.0
    return min_area


def _hull_area(pts):
    """Return the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area. Higher is better."""
    if points is None:
        return -1.0
    pts = np.asarray(points, dtype=float)
    if pts.shape != (N, 2):
        return -1.0
    ha = _hull_area(pts)
    if ha < 1e-15:
        return -1.0
    mt = _min_triangle_area(pts)
    return mt / ha


def _decode(x):
    """Decode a flat vector of length 2*N into N points."""
    return x.reshape(N, 2)


def _regular_polygon():
    """Regular N-gon inscribed in unit circle – a strong symmetric baseline."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def construct():
    start = time.time()

    # Baseline: regular polygon
    best_pts = _regular_polygon()
    best_score = _score(best_pts)

    # Try a few symmetric perturbations / different starting points
    # Differential evolution on the full 2N coordinate space
    # Objective: maximize score => minimize -score
    def objective(x):
        pts = _decode(x)
        return -_score(pts)

    # Bounds: unit square [0,1]^2
    bounds = [(0.0, 1.0)] * (2 * N)

    # Run differential evolution with time-limited budget
    elapsed = time.time() - start
    if elapsed < 60:  # leave time for polish
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=200,
                tol=1e-6,
                seed=42,
                polish=False,
                popsize=10,
                init='sobol',
                callback=_de_callback,
            )
            elapsed = time.time() - start
            if elapsed < 100:
                cand = _decode(result.x)
                sc = _score(cand)
                if sc > best_score:
                    best_score = sc
                    best_pts = cand
        except Exception:
            pass

    # Local polish: coordinate ascent / small random perturbations around best
    elapsed = time.time() - start
    if elapsed < 105:
        step = 0.05
        for _ in range(500):
            if time.time() - start > 110:
                break
            pts = best_pts + step * np.random.randn(N, 2)
            pts = np.clip(pts, 0, 1)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts

    # Final high-precision: small random search around best
    elapsed = time.time() - start
    if elapsed < 110:
        for _ in range(2000):
            if time.time() - start > 114:
                break
            pts = best_pts + 0.01 * np.random.randn(N, 2)
            pts = np.clip(pts, 0, 1)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts

    return best_pts


def _de_callback(xk, convergence=False):
    if time.time() - _DE_START > 95:
        raise _TimeoutException()


class _TimeoutException(Exception):
    pass


# We need _DE_START to be accessible; set it at start of construct
# Actually, let's restructure to avoid the callback complexity.
# Simpler: just use maxiter and rely on scipy's internal stopping.
# Rewrite construct without callback:

def construct():
    start = time.time()

    # Baseline: regular polygon
    best_pts = _regular_polygon()
    best_score = _score(best_pts)

    # Differential evolution
    def objective(x):
        pts = x.reshape(N, 2)
        return -_score(pts)

    bounds = [(0.0, 1.0)] * (2 * N)
    elapsed = time.time() - start
    if elapsed < 80:
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=150,
                tol=1e-5,
                seed=42,
                polish=True,
                popsize=12,
                init='sobol',
            )
            elapsed = time.time() - start
            cand = _decode(result.x)
            sc = _score(cand)
            if sc > best_score:
                best_score = sc
                best_pts = cand
        except Exception:
            pass

    # Coordinate ascent polish
    elapsed = time.time() - start
    step = 0.05
    for _ in range(1000):
        if time.time() - start > 112:
            break
        for i in range(N):
            for d in range(2):
                if time.time() - start > 112:
                    break
                orig = best_pts[i, d]
                for delta in [step, -step, step * 0.5, -step * 0.5]:
                    trial = best_pts.copy()
                    trial[i, d] = np.clip(orig + delta, 0, 1)
                    sc = _score(trial)
                    if sc > best_score:
                        best_score = sc
                        best_pts = trial
                        break

    # Fine random search
    elapsed = time.time() - start
    for _ in range(3000):
        if time.time() - start > 114:
            break
        pts = best_pts + 0.005 * np.random.randn(N, 2)
        pts = np.clip(pts, 0, 1)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts

    return best_pts
