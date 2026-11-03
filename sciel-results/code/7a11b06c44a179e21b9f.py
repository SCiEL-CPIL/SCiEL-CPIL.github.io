"""Model-and-optimize approach for the Heilbronn-11 triangle problem."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 142  # safety margin from 150s limit

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _bary_to_cart(bary):
    """Convert barycentric coords (N,3) to Cartesian (N,2)."""
    # bary[:,0]*V0 + bary[:,1]*V1 + bary[:,2]*V2
    # V0=(0,0), V1=(1,0), V2=(0.5, S3H)
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
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _active_soft_min(pts, active_idx, beta):
    """Soft min over a subset of triangle indices."""
    pi = pts[_T_I[active_idx]]
    pj = pts[_T_J[active_idx]]
    pk = pts[_T_K[active_idx]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_bary(bary):
    """Project barycentric coordinates to be valid (all >= 0, sum = 1)."""
    bary = bary.copy()
    # Clip negatives
    bary = np.maximum(bary, 0.0)
    # Normalize
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.maximum(sums, 1e-15)
    bary = bary / sums
    return bary


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
    # Use a triangular lattice with K=4 divisions (16 points, pick 11 best)
    K = 4
    candidates = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            # Barycentric: (1 - i/K - j/K, i/K, j/K)
            b = np.array([1.0 - i/K - j/K, i/K, j/K])
            cart = _bary_to_cart(b.reshape(1, 3))[0]
            candidates.append(cart)
    candidates = np.array(candidates)
    
    # Greedy selection: pick points that maximize min distance
    init_pts = np.zeros((N, 2))
    # Start with corners
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
            # Score: min distance to already selected
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

    def _random_config():
        """Generate a random feasible configuration."""
        pts = np.array([_random_in_triangle() for _ in range(N)])
        return pts

    def _perturb_bary(pts, scale):
        """Perturb points in barycentric space and project back."""
        bary = _cart_to_bary(pts)
        bary = bary + rng.normal(0, scale, size=(N, 3))
        bary = _project_bary(bary)
        return _bary_to_cart(bary)

    def _optimize_bary(pts0, beta, maxiter=200):
        """Optimize in barycentric space using L-BFGS-B."""
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

    def _active_set_optimize(pts0, beta, max_iters=10):
        """Active-set optimization: only optimize against current min triangles."""
        pts = pts0.copy()
        for it in range(max_iters):
            if not _elapsed_ok(2):
                break
            # Find current min triangle
            min_area, min_idx = _min_triangle_area_and_idx(pts)
            
            # Get active set: triangles involving the 3 vertices of min triangle
            tri_verts = _TRIPLETS[min_idx]
            # Find all triangles that share at least one vertex with the min triangle
            active_mask = np.zeros(len(_TRIPLETS), dtype=bool)
            for v in tri_verts:
                active_mask |= (_T_I == v) | (_T_J == v) | (_T_K == v)
            active_idx = np.where(active_mask)[0]
            
            # Also include the global min
            active_idx = np.unique(np.concatenate([active_idx, [min_idx]]))
            
            # Optimize using active set
            bary0 = _cart_to_bary(pts)
            
            def objective(bary_flat):
                b = _project_bary(bary_flat.reshape(-1, 3))
                p = _bary_to_cart(b)
                return -_active_soft_min(p, active_idx, beta)
            
            try:
                res = minimize(objective, bary0.flatten(), method='L-BFGS-B',
                               options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10})
                b = _project_bary(res.x.reshape(-1, 3))
                pts = _bary_to_cart(b)
            except Exception:
                break
            
            _update_best(pts)
            
            # Check if topology changed
            new_min_area, new_min_idx = _min_triangle_area_and_idx(pts)
            if abs(new_min_area - min_area) < 1e-12:
                break
        
        return pts

    # ---- Phase A: Multi-start with barycentric optimization ----
    n_starts = 30
    bases = [init_pts]
    for _ in range(15):
        bases.append(_perturb_bary(init_pts, 0.3))

    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        base = bases[s % len(bases)]
        pts = _perturb_bary(base, 0.08)
        _update_best(pts)
        for beta in [15.0, 50.0, 150.0, 400.0]:
            if not _elapsed_ok(30):
                break
            pts = _optimize_bary(pts, beta, maxiter=150)
            _update_best(pts)

    # ---- Phase B: Active-set refinement on best ----
    if best_pts is not None and _elapsed_ok(15):
        for _ in range(3):
            if not _elapsed_ok(15):
                break
            pts = _active_set_optimize(best_pts, beta=100, max_iters=8)
            _update_best(pts)

    # ---- Phase C: Nelder-Mead local polish on the best configuration ----
    if best_pts is not None and _elapsed_ok(10):
        for scale in [0.04, 0.02, 0.01]:
            if not _elapsed_ok(8):
                break
            for _ in range(5):
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
                                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
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
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            b = _project_bary(res.x.reshape(-1, 3))
            p = _bary_to_cart(b)
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        # fallback random feasible configuration
        pts = np.array([_random_in_triangle() for _ in range(N)])
        return pts

    # Final: ensure validity via barycentric round-trip
    bary = _cart_to_bary(best_pts)
    bary = _project_bary(bary)
    best_pts = _bary_to_cart(bary)
    return best_pts
# EVOLVE-BLOCK-END
