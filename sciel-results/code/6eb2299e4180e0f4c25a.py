"""Optimize 14 points in unit-area convex region to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = time.time() + 140  # safety margin before 150s kill
_TIME_BUDGET = 130  # soft budget for search


def _min_triangle_area(pts):
    """Minimum triangle area over all C(n,3) triples."""
    n = len(pts)
    min_a = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) -
                              (pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1]))
                if a < min_a:
                    min_a = a
    return min_a


def _convex_hull_area(pts):
    """Area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """min triangle area / convex hull area."""
    ch = _convex_hull_area(pts)
    if ch < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ch


def _smooth_neg_score(pts, beta=50.0):
    """Smooth surrogate: -soft_min of triangle areas / hull area.
    Using log-sum-exp for soft-min."""
    n = len(pts)
    # Compute all triangle areas vectorized-ish
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) -
                              (pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1]))
                areas.append(a)
    areas = np.array(areas)
    if areas.max() < 1e-15:
        return 0.0
    # soft-min via log-sum-exp: soft_min = -log(sum(exp(-beta*a))) / beta
    # For numerical stability:
    m = areas.min()
    soft_min = m - np.log(np.sum(np.exp(-beta*(areas - m)))) / beta
    ch = _convex_hull_area(pts)
    if ch < 1e-12:
        return 0.0
    return -soft_min / ch


def _regular_grid_init(seed=42):
    """Generate a reasonable starting configuration."""
    rng = np.random.RandomState(seed)
    # Start with points on a circle (good for max-min area)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.uniform(0, 0.1, N)
    radii = np.ones(N) * (1.0 + rng.uniform(-0.05, 0.05, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _polish(pts, beta=100.0, max_iter=200):
    """Local polish using smooth surrogate with gradient-based optimizer."""
    def obj(x):
        p = x.reshape(N, 2)
        return -_smooth_neg_score(p, beta=beta)
    
    try:
        res = minimize(obj, pts.ravel(), method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': 1e-12})
        p = res.x.reshape(N, 2)
        return p
    except Exception:
        return pts


def _differential_evolution_search(bounds, max_iter=50, popsize=15):
    """Run differential evolution on the smooth surrogate."""
    def obj(x):
        p = x.reshape(N, 2)
        return -_smooth_neg_score(p, beta=30.0)
    
    try:
        res = differential_evolution(obj, bounds, maxiter=max_iter,
                                      popsize=popsize, tol=1e-6,
                                      seed=42, polish=True,
                                      workers=1)
        return res.x.reshape(N, 2)
    except Exception:
        return None


def construct():
    global _DEADLINE
    start = time.time()
    
    best_pts = _regular_grid_init(42)
    best_score = _score(best_pts)
    
    # Strategy 1: Differential evolution with smooth surrogate
    if time.time() < _DEADLINE - 20:
        # Normalize to roughly unit area region
        bounds = [(-1, 1)] * (2 * N)
        de_pts = _differential_evolution_search(bounds, max_iter=30, popsize=12)
        if de_pts is not None:
            s = _score(de_pts)
            if s > best_score:
                best_score = s
                best_pts = de_pts
    
    # Strategy 2: Polish from multiple initializations
    for seed in range(10):
        if time.time() > _DEADLINE - 10:
            break
        init_pts = _regular_grid_init(seed)
        polished = _polish(init_pts, beta=80.0, max_iter=100)
        s = _score(polished)
        if s > best_score:
            best_score = s
            best_pts = polished
    
    # Strategy 3: High-precision polish on best
    if time.time() < _DEADLINE - 5:
        final = _polish(best_pts, beta=200.0, max_iter=300)
        s = _score(final)
        if s > best_score:
            best_score = s
            best_pts = final
    
    # Normalize: scale so convex hull area = 1
    ch = _convex_hull_area(best_pts)
    if ch > 1e-12:
        scale = 1.0 / np.sqrt(ch)
        best_pts = best_pts * scale
    
    return best_pts
