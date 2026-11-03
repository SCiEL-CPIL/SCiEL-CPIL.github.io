"""Delaunay-based active-set optimization for the Heilbronn-11 triangle problem."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import Delaunay

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 138.0  # safety margin from 150s limit

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])
VERTS = np.array([V0, V1, V2])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165


def _bary_to_cart(bary):
    """Convert barycentric coords (N,3) to Cartesian (N,2)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack([x, y])


def _cart_to_bary(cart):
    """Convert Cartesian (N,2) to barycentric coords (N,3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3H
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _project_bary(bary):
    """Project barycentric coordinates to be valid (all >= 0, sum = 1)."""
    bary = bary.copy()
    bary = np.maximum(bary, 0.0)
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.maximum(sums, 1e-15)
    bary = bary / sums
    return bary


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _min_triangle_area_and_idx(pts):
    """Compute the minimum area and its index."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    idx = np.argmin(areas)
    return areas[idx], idx


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_a - log_sum / beta
    return soft_min


def _delaunay_triangles(pts):
    """Get the indices of Delaunay triangles (triplets of point indices)."""
    try:
        tri = Delaunay(pts)
        # tri.simplices has shape (num_tris, 3)
        # Each row is a triplet of point indices
        return tri.simplices
    except Exception:
        return None


def _delaunay_min_area(pts):
    """Compute minimum area among Delaunay triangles only."""
    simplices = _delaunay_triangles(pts)
    if simplices is None or len(simplices) == 0:
        return _min_triangle_area(pts)
    
    areas = []
    for s in simplices:
        i, j, k = s
        # Ensure sorted order to match our triplet convention
        s_sorted = tuple(sorted(s))
        p = pts[s]
        dx1 = p[1, 0] - p[0, 0]
        dy1 = p[1, 1] - p[0, 1]
        dx2 = p[2, 0] - p[0, 0]
        dy2 = p[2, 1] - p[0, 1]
        cross = dx1 * dy2 - dy1 * dx2
        areas.append(abs(cross) * 0.5)
    
    if not areas:
        return _min_triangle_area(pts)
    
    return min(areas)


def _get_active_set(pts, n_active=15):
    """Get active set of triangle indices: Delaunay triangles + smallest non-Delaunay."""
    # First, get all triangle areas
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Get Delaunay triangle indices
    simplices = _delaunay_triangles(pts)
    delaunay_idx = set()
    if simplices is not None:
        for s in simplices:
            s_sorted = tuple(sorted(s))
            # Find the index in _TRIPLETS
            for idx, t in enumerate(_TRIPLETS):
                if t == s_sorted:
                    delaunay_idx.add(idx)
                    break
    
    # Get indices sorted by area
    sorted_idx = np.argsort(areas)
    
    # Active set: all Delaunay + smallest non-Delaunay to fill up to n_active
    active = list(delaunay_idx)
    for idx in sorted_idx:
        if len(active) >= n_active:
            break
        if idx not in delaunay_idx:
            active.append(idx)
    
    return np.array(active, dtype=int)


def _active_soft_min(pts, active_idx, beta):
    """Soft min over a subset of triangle indices."""
    if len(active_idx) == 0:
        return _soft_min_area(pts, beta)
    pi = pts[_T_I[active_idx]]
    pj = pts[_T_J[active_idx]]
    pk = pts[_T_K[active_idx]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_a - log_sum / beta
    return soft_min


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Better initialization: denser grid-based points ----
    K = 4
    candidates = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            b = np.array([1.0 - i/K - j/K, i/K, j/K])
            cart = _bary_to_cart(b.reshape(1, 3))[0]
            candidates.append(cart)
    candidates = np.array(candidates)

    # Greedy selection: pick points that maximize min distance
    init_pts = np.zeros((N, 2))
    init_pts[0] = V0
    init_pts[1] = V1
    init_pts[2] = V2
    selected = [0, 1, 2]
    for k in range(3, N):
        best_score = -1
        best_idx = -1
        for ci in range(len(candidates)):
            if ci in selected:
                continue
            cand = candidates[ci]
            dists = np.linalg.norm(init_pts[:k] - cand, axis=1)
            score = np.min(dists)
            if score > best_score:
                best_score = score
                best_idx = ci
        init_pts[k] = candidates[best_idx]
        selected.append(best_idx)

    _update_best(init_pts)

    def _random_in_triangle():
        """Generate a random point in the triangle using barycentric coords."""
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        bary = np.array([1 - u - v, u, v])
        return _bary_to_cart(bary.reshape(1, 3))[0]

    def _perturb_bary(pts, scale):
        """Perturb points in barycentric space and project back."""
        bary = _cart_to_bary(pts)
        bary = bary + rng.normal(0, scale, size=(N, 3))
        bary = _project_bary(bary)
        return _bary_to_cart(bary)

    def _delaunay_optimize(pts0, beta, max_iters=15):
        """Delaunay-based active-set optimization."""
        pts = pts0.copy()
        for it in range(max_iters):
            if not _elapsed_ok(3):
                break
            
            # Get current active set
            active_idx = _get_active_set(pts, n_active=15)
            
            # Optimize using active set
            bary0 = _cart_to_bary(pts)
            
            def objective(bary_flat):
                b = _project_bary(bary_flat.reshape(-1, 3))
                p = _bary_to_cart(b)
                return -_active_soft_min(p, active_idx, beta)
            
            try:
                res = minimize(objective, bary0.flatten(), method='L-BFGS-B',
                               options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10})
                b = _project_bary(res.x.reshape(-1, 3))
                new_pts = _bary_to_cart(b)
            except Exception:
                break
            
            new_area = _min_triangle_area(new_pts)
            if new_area > _min_triangle_area(pts):
                pts = new_pts
                _update_best(pts)
            else:
                break
        
        return pts

    def _full_optimize(pts0, beta, maxiter=200):
        """Full optimization using all 165 triangles."""
        bary0 = _cart_to_bary(pts0)
        
        def objective(bary_flat):
            b = _project_bary(bary_flat.reshape(-1, 3))
            pts = _bary_to_cart(b)
            return -_soft_min_area(pts, beta)
        
        try:
            res = minimize(objective, bary0.flatten(), method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            b = _project_bary(res.x.reshape(-1, 3))
            return _bary_to_cart(b)
        except Exception:
            return pts0

    # ---- Phase A: Multi-start with Delaunay active-set optimization ----
    n_starts = 25
    bases = [init_pts]
    for _ in range(12):
        bases.append(_perturb_bary(init_pts, 0.3))

    for s in range(n_starts):
        if not _elapsed_ok(25):
            break
        base = bases[s % len(bases)]
        pts = _perturb_bary(base, 0.08)
        _update_best(pts)
        
        # First pass: Delaunay active-set with increasing beta
        for beta in [20.0, 60.0, 150.0, 300.0]:
            if not _elapsed_ok(25):
                break
            pts = _delaunay_optimize(pts, beta, max_iters=12)
            _update_best(pts)
        
        # Second pass: full optimization with high beta
        if _elapsed_ok(25):
            pts = _full_optimize(pts, beta=400.0, maxiter=100)
            _update_best(pts)

    # ---- Phase B: Delaunay refinement on best ----
    if best_pts is not None and _elapsed_ok(15):
        for _ in range(4):
            if not _elapsed_ok(15):
                break
            pts = _delaunay_optimize(best_pts, beta=200, max_iters=10)
            _update_best(pts)
            # Also try a full optimization pass
            if _elapsed_ok(15):
                pts = _full_optimize(pts, beta=500.0, maxiter=80)
                _update_best(pts)

    # ---- Phase C: Nelder-Mead local polish on the best configuration ----
    if best_pts is not None and _elapsed_ok(10):
        for scale in [0.03, 0.015, 0.008]:
            if not _elapsed_ok(8):
                break
            for _ in range(4):
                if not _elapsed_ok(8):
                    break
                cand = _perturb_bary(best_pts, scale)

                def nm_obj(x):
                    b = _project_bary(x.reshape(-1, 3))
                    p = _bary_to_cart(b)
                    return -_min_triangle_area(p)
                try:
                    bary0 = _cart_to_bary(cand).flatten()
                    res = minimize(nm_obj, bary0, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                    b = _project_bary(res.x.reshape(-1, 3))
                    p = _bary_to_cart(b)
                    _update_best(p)
                except Exception:
                    pass

    # ---- Final fine polish on the absolute best ----
    if best_pts is not None and _elapsed_ok(5):
        def nm_obj(x):
            b = _project_bary(x.reshape(-1, 3))
            p = _bary_to_cart(b)
            return -_min_triangle_area(p)
        try:
            bary0 = _cart_to_bary(best_pts).flatten()
            res = minimize(nm_obj, bary0, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
            b = _project_bary(res.x.reshape(-1, 3))
            p = _bary_to_cart(b)
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        pts = np.array([_random_in_triangle() for _ in range(N)])
        return pts

    # Final: ensure validity via barycentric round-trip
    bary = _cart_to_bary(best_pts)
    bary = _project_bary(bary)
    best_pts = _bary_to_cart(bary)
    return best_pts
# EVOLVE-BLOCK-END
