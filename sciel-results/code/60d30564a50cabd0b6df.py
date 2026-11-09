# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
import itertools

_DEADLINE = time.time() + 600.0

def _min_triangle_area(points):
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _optimize_interior(outer, inner0, max_sweeps=400, n_inner=None):
    """
    Cyclic coordinate ascent on interior points to maximize min triangle area.
    Outer ring is fixed. inner is a (k,2) array.
    """
    if n_inner is None:
        n_inner = len(inner0)
    inner = inner0.copy()
    outer_arr = np.asarray(outer)
    best_min = _min_triangle_area(np.vstack([outer_arr, inner]))
    best_inner = inner.copy()
    
    # We only care about min area; hull area is fixed (outer ring), so maximizing min area maximizes the ratio.
    # Actually hull area might change if an interior point pokes out, but we keep interior strictly inside.
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 10:
            break
        improved = False
        for i in range(n_inner):
            # Optimize point i's x and y via small local search
            x0, y0 = inner[i]
            best_x, best_y = x0, y0
            best_val = best_min  # current global min area (approx, but we recompute per candidate)
            
            # Radial search: try a few step sizes
            for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                               (step*0.707,step*0.707),(-step*0.707,step*0.707),
                               (step*0.707,-step*0.707),(-step*0.707,-step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    # Keep strictly inside the unit square and inside outer ring
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    new_inner = inner.copy()
                    new_inner[i] = [nx, ny]
                    all_pts = np.vstack([outer_arr, new_inner])
                    # Check all points are inside or on hull (they should be if interior)
                    try:
                        ma = _min_triangle_area(all_pts)
                    except Exception:
                        continue
                    if ma > best_val + 1e-12:
                        best_val = ma
                        best_x, best_y = nx, ny
                        improved = True
            inner[i] = [best_x, best_y]
        
        # After a full sweep, update global best
        all_pts = np.vstack([outer_arr, inner])
        cur_min = _min_triangle_area(all_pts)
        if cur_min > best_min + 1e-12:
            best_min = cur_min
            best_inner = inner.copy()
        
        # Early stop if no improvement
        if not improved:
            break
    
    return best_inner, best_min

def _optimize_full_ratio(points0, max_sweeps=300):
    """
    More general: optimize all 13 points via coordinate ascent on min-area,
    but periodically re-normalize to keep them in a good region.
    """
    pts = points0.copy()
    best_min = _min_triangle_area(pts)
    best_pts = pts.copy()
    
    n = len(pts)
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 15:
            break
        improved = False
        for i in range(n):
            x0, y0 = pts[i]
            best_x, best_y = x0, y0
            best_val = best_min
            for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                               (step*0.707,step*0.707),(-step*0.707,step*0.707),
                               (step*0.707,-step*0.707),(-step*0.707,-step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    ma = _min_triangle_area(new_pts)
                    if ma > best_val + 1e-12:
                        best_val = ma
                        best_x, best_y = nx, ny
                        improved = True
            pts[i] = [best_x, best_y]
        
        cur_min = _min_triangle_area(pts)
        if cur_min > best_min + 1e-12:
            best_min = cur_min
            best_pts = pts.copy()
        if not improved:
            break
    
    return best_pts, best_min

def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=12345)
    
    best_points = None
    best_score = -np.inf
    
    # Strategy: Fix outer ring as regular polygon, optimize interior.
    # Try different splits: k outer, (13-k) inner
    # Best known: for Heilbronn, having more on the hull is generally good.
    # Try k_outer in [9, 10, 11, 12, 13]
    
    for k_outer in [13, 12, 11, 10, 9]:
        if time.time() > _DEADLINE - 60:
            break
        k_inner = n - k_outer
        
        # Generate outer ring
        outer = _regular_polygon(k_outer, radius=0.48, center=(0.5, 0.5))
        
        if k_inner == 0:
            pts = outer
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
            continue
        
        # Generate initial interior points: center + small ring
        if k_inner == 1:
            inner0 = np.array([[0.5, 0.5]])
        elif k_inner == 2:
            inner0 = np.array([[0.5 + 0.15, 0.5], [0.5 - 0.15, 0.5]])
        elif k_inner == 3:
            angles = np.linspace(0, 2*np.pi, 3, endpoint=False)
            inner0 = np.column_stack([0.5 + 0.2*np.cos(angles), 0.5 + 0.2*np.sin(angles)])
        else:
            angles = np.linspace(0, 2*np.pi, k_inner, endpoint=False)
            inner0 = np.column_stack([0.5 + 0.2*np.cos(angles), 0.5 + 0.2*np.sin(angles)])
        
        # Optimize interior
        inner_opt, min_area = _optimize_interior(outer, inner0, max_sweeps=300)
        pts = np.vstack([outer, inner_opt])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        
        # Also try a few random interior starts
        for trial in range(3):
            if time.time() > _DEADLINE - 30:
                break
            inner_rand = rng.uniform(0.3, 0.7, (k_inner, 2))
            inner_opt2, _ = _optimize_interior(outer, inner_rand, max_sweeps=150)
            pts2 = np.vstack([outer, inner_opt2])
            s2 = _score(pts2)
            if s2 > best_score:
                best_score = s2
                best_points = pts2.copy()
    
    # Also try full coordinate ascent from a regular 13-gon
    if time.time() < _DEADLINE - 60:
        pts13 = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
        pts_opt, _ = _optimize_full_ratio(pts13, max_sweeps=200)
        s = _score(pts_opt)
        if s > best_score:
            best_score = s
            best_points = pts_opt.copy()
    
    # Also try full coordinate ascent from a ring+interior start
    if time.time() < _DEADLINE - 40:
        outer9 = _regular_polygon(9, radius=0.48, center=(0.5, 0.5))
        angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        inner4 = np.column_stack([0.5 + 0.2*np.cos(angles4), 0.5 + 0.2*np.sin(angles4)])
        start = np.vstack([outer9, inner4])
        pts_opt2, _ = _optimize_full_ratio(start, max_sweeps=200)
        s2 = _score(pts_opt2)
        if s2 > best_score:
            best_score = s2
            best_points = pts_opt2.copy()
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END
