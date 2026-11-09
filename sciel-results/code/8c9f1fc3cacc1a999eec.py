# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _all_triangles(n):
    """Precompute all C(n,3) triangle index triples."""
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)


def _triangle_areas(pts, tri_idx):
    """Compute areas of all triangles given points and index triples."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    # area = 0.5*|x1(y2-y3)+x2(y3-y1)+x3(y1-y2)|
    area = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1]) +
                        b[:, 0] * (c[:, 1] - a[:, 1]) +
                        c[:, 0] * (a[:, 1] - b[:, 1]))
    return area


def _hull_area(pts):
    """Convex hull area; 0 on failure."""
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(pts):
    """Compute A_min / A_hull."""
    n = len(pts)
    tri_idx = _all_triangles(n)
    areas = _triangle_areas(pts, tri_idx)
    a_min = float(areas.min())
    a_hull = _hull_area(pts)
    if a_hull <= 0:
        return 0.0
    return a_min / a_hull


def _subgradient(pts, tri_idx):
    """
    Compute subgradient of the minimum triangle area with respect to points.
    Returns gradient array of shape (n, 2).
    """
    n = len(pts)
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    
    # Find all triangles achieving (near) the minimum
    tol = 1e-10
    active = np.where(np.abs(areas - min_area) < tol)[0]
    
    grad = np.zeros((n, 2))
    for t in active:
        i, j, k = tri_idx[t]
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        # area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
        # Let cross = x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)
        # area = 0.5 * |cross|
        # d(area)/d(pi) = 0.5 * sign(cross) * (y_j - y_k, x_k - x_j)
        # d(area)/d(pj) = 0.5 * sign(cross) * (y_k - y_i, x_i - x_k)
        # d(area)/d(pk) = 0.5 * sign(cross) * (y_i - y_j, x_j - x_i)
        
        cross = pi[0] * (pj[1] - pk[1]) + pj[0] * (pk[1] - pi[1]) + pk[0] * (pi[1] - pj[1])
        s = np.sign(cross)
        if s == 0:
            s = 1.0
        
        dpi = 0.5 * s * np.array([pj[1] - pk[1], pk[0] - pj[0]])
        dpj = 0.5 * s * np.array([pk[1] - pi[1], pi[0] - pk[0]])
        dpk = 0.5 * s * np.array([pi[1] - pj[1], pj[0] - pi[0]])
        
        grad[i] += dpi
        grad[j] += dpj
        grad[k] += dpk
    
    # Normalize gradient
    norm = np.linalg.norm(grad)
    if norm > 0:
        grad /= norm
    return grad


def _gradient_ascent(pts, tri_idx, max_iter=2000, lr_init=0.05, lr_decay=0.999, deadline=None):
    """
    Perform subgradient ascent on the minimum triangle area.
    """
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    lr = lr_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        grad = _subgradient(pts, tri_idx)
        pts_new = pts + lr * grad
        
        # Ensure points stay finite
        if not np.all(np.isfinite(pts_new)):
            break
        
        score = _score(pts_new)
        if score > best_score:
            best_score = score
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # Reduce learning rate
            lr *= 0.95
            if lr < 1e-6:
                break
        
        # Check convergence
        if it > 100 and abs(score - best_score) < 1e-12:
            break
    
    return best_pts, best_score


def _regular_polygon_init(n, n_sides=12, center_weight=1.0):
    """Initialize with regular polygon vertices + center."""
    angles = 2 * np.pi * np.arange(n_sides) / n_sides
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Add center
    pts = np.vstack([pts, np.array([[0.0, 0.0]])])
    # Shuffle to mix
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n)
    return pts[perm]


def _grid_init(n):
    """Initialize with a grid-like pattern."""
    rng = np.random.default_rng(seed=42)
    # 4x4 grid minus one point, then jitter
    grid = []
    for i in range(4):
        for j in range(4):
            grid.append([i - 1.5, j - 1.5])
    grid = np.array(grid[:n])
    # Add small random jitter
    grid += rng.random((n, 2)) * 0.1
    return grid


def _random_init(n, seed=42):
    """Random initialization in unit square."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.
    
    Returns:
        points: np.ndarray of shape (13,2)
    """
    n = 13
    tri_idx = _all_triangles(n)
    
    deadline = time.time() + 600
    
    # Multiple initializations
    inits = []
    inits.append(('polygon', _regular_polygon_init(n)))
    inits.append(('grid', _grid_init(n)))
    
    # Several random inits with different seeds
    for s in [0, 1, 2, 3, 4]:
        inits.append((f'rand{s}', _random_init(n, seed=s)))
    
    best_pts = None
    best_score = 0.0
    
    for name, pts in inits:
        if time.time() > deadline - 30:
            break
        
        # Run gradient ascent
        result_pts, result_score = _gradient_ascent(
            pts, tri_idx, 
            max_iter=1500, 
            lr_init=0.03, 
            lr_decay=0.999,
            deadline=deadline
        )
        
        if result_score > best_score:
            best_score = result_score
            best_pts = result_pts.copy()
    
    if best_pts is None:
        # Fallback: return random points
        best_pts = _random_init(n, seed=42)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _random_init(n, seed=42)
    
    return best_pts


# EVOLVE-BLOCK-END
