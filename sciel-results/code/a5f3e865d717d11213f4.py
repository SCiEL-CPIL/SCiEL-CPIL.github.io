"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14

# Deadline guard: return best before 145s (safety margin under 150s)
_START_TIME = time.time()
TIME_LIMIT = 145.0

def _min_triangle_area(pts):
    """Minimum triangle area over all C(n,3) triples. Vectorized."""
    n = pts.shape[0]
    min_a = np.inf
    # Vectorize over pairs (i,j) and compute cross product with each k
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v = pts[j] - pts[i]
            # cross product with all other points
            d = pts - pts[i]
            cross = v[0] * d[:, 1] - v[1] * d[:, 0]
            # only consider k > j
            areas = 0.5 * np.abs(cross[j + 1:])
            if areas.size > 0:
                m = np.min(areas)
                if m < min_a:
                    min_a = m
    return min_a

def _convex_hull_area(pts):
    """Area of convex hull using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume = area
    except Exception:
        # Fallback: compute via shoelace on sorted angular points
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(angles)
        hp = pts[order]
        x = hp[:, 0]
        y = hp[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _score(pts):
    """min triangle area / convex hull area."""
    mta = _min_triangle_area(pts)
    cha = _convex_hull_area(pts)
    if cha <= 0:
        return 0.0
    return mta / cha

def _soft_min_triangle_area(pts, temperature=0.01):
    """Soft surrogate: log-sum-exp of -areas, gives smooth min."""
    n = pts.shape[0]
    neg_sum = 0.0
    min_a = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v = pts[j] - pts[i]
            d = pts - pts[i]
            cross = v[0] * d[:, 1] - v[1] * d[:, 0]
            areas = 0.5 * np.abs(cross[j + 1:])
            if areas.size == 0:
                continue
            # soft min via log-sum-exp of -areas
            max_neg = -np.min(areas)  # = -min_area
            exps = np.exp(-(areas - np.min(areas)) / temperature)
            soft_min = np.min(areas) + temperature * np.log(np.sum(exps))
            if soft_min < min_a:
                min_a = soft_min
    return min_a

def _neg_score_smooth(pts, temperature=0.02):
    """Negative smooth score for minimization."""
    mta = _soft_min_triangle_area(pts, temperature)
    cha = _convex_hull_area(pts)
    if cha <= 1e-12:
        return 1e6
    return -mta / cha

def _make_concentric(k_outer, k_inner):
    """Create concentric regular polygons: k_outer on unit circle, k_inner on inner circle."""
    pts = []
    # Outer ring: regular k_outer-gon on circle radius 1
    for i in range(k_outer):
        angle = 2 * np.pi * i / k_outer
        pts.append([np.cos(angle), np.sin(angle)])
    # Inner ring: regular k_inner-gon on circle radius r_inner, rotated to interleave
    r_inner = 0.5  # will be optimized
    offset = np.pi / k_inner
    for i in range(k_inner):
        angle = 2 * np.pi * i / k_inner + offset
        pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
    return np.array(pts)

def _make_grid_like():
    """A grid-like initial configuration."""
    pts = []
    # 4x4 grid perturbed slightly, scaled to fit in unit area region
    for i in range(4):
        for j in range(4):
            x = (i - 1.5) / 1.5
            y = (j - 1.5) / 1.5
            pts.append([x, y])
    return np.array(pts[:14])

def _make_fibonacci():
    """Fibonacci sphere projection to 2D."""
    pts = []
    golden = (1 + np.sqrt(5)) / 2
    for i in range(N):
        angle = 2 * np.pi * i / golden
        r = np.sqrt((i + 0.5) / N)
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(pts)

def _refine(pts0, max_time=30.0, temperature=0.02):
    """Refine a starting configuration using L-BFGS-B on smooth objective."""
    start = time.time()
    x0 = pts0.flatten()
    
    def objective(x):
        if time.time() - start > max_time:
            return 1e6
        pts = x.reshape(-1, 2)
        return _neg_score_smooth(pts, temperature)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        best_pts = res.x.reshape(-1, 2)
        best_score = _score(best_pts)
    except Exception:
        best_pts = pts0
        best_score = _score(pts0)
    
    return best_pts, best_score

def construct():
    global _START_TIME
    if _START_TIME == 0:
        _START_TIME = time.time()
    
    best_pts = None
    best_score = -np.inf
    
    # Generate multiple initial configurations
    inits = []
    
    # Concentric: 7 outer + 7 inner
    inits.append(_make_concentric(7, 7))
    # Concentric: 8 outer + 6 inner  
    inits.append(_make_concentric(8, 6))
    # Concentric: 6 outer + 8 inner
    inits.append(_make_concentric(6, 8))
    # Concentric: 14 on one circle (regular 14-gon)
    inits.append(_make_concentric(14, 0))
    # Fibonacci
    inits.append(_make_fibonacci())
    # Grid-like
    inits.append(_make_grid_like())
    
    # Try different inner radii for concentric configs
    for r in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pts = _make_concentric(7, 7)
        # Scale inner points
        for i in range(7, 14):
            pts[i] = r * pts[i] / 0.5
        inits.append(pts)
    
    # Refine each initial configuration
    elapsed = time.time() - _START_TIME
    time_per_init = max(1.0, (TIME_LIMIT - 5.0 - elapsed) / len(inits))
    
    for idx, pts0 in enumerate(inits):
        if time.time() - _START_TIME > TIME_LIMIT - 10.0:
            break
        
        remaining = TIME_LIMIT - 5.0 - (time.time() - _START_TIME)
        if remaining < 2.0:
            break
        
        alloc_time = min(time_per_init, remaining)
        pts_refined, score = _refine(pts0, max_time=alloc_time, temperature=0.02)
        
        if score > best_score:
            best_score = score
            best_pts = pts_refined
    
    # Final polish: one more refinement with tighter tolerance on best found
    if best_pts is not None and time.time() - _START_TIME < TIME_LIMIT - 5.0:
        remaining = TIME_LIMIT - 5.0 - (time.time() - _START_TIME)
        if remaining > 3.0:
            pts_polished, score_polished = _refine(best_pts, max_time=remaining, temperature=0.005)
            if score_polished > best_score:
                best_pts = pts_polished
                best_score = score_polished
    
    # Safety: if no valid result, return a fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        # Regular 14-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize: scale so convex hull area = 1
    cha = _convex_hull_area(best_pts)
    if cha > 1e-12:
        best_pts = best_pts / np.sqrt(cha)
    
    return best_pts

# Initialize start time
_START_TIME = time.time()
