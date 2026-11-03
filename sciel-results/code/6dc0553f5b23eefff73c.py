"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from itertools import combinations

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 145  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)


def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    return np.min(_all_triangle_areas(pts))


def _delaunay_triangle_indices(pts):
    """
    Get the indices of triangles from Delaunay triangulation of pts.
    Returns a list of (i, j, k) tuples where i,j,k are indices into pts.
    Only includes triangles fully inside the unit equilateral triangle.
    """
    try:
        tri = Delaunay(pts)
        # tri.simplices has shape (n_triangles, 3)
        # We need to filter to only triangles that are "inside" our domain
        # A triangle is inside if all its vertices are inside the triangle
        # Since all our points are inside, we just need to check that the triangle
        # formed by three points doesn't go outside... actually for Delaunay of points
        # inside a convex region, the triangles are the convex hull triangles.
        # We'll just use all simplices from Delaunay.
        return [tuple(s) for s in tri.simplices]
    except Exception:
        return []


def _delaunay_min_area(pts):
    """
    Compute the minimum area among Delaunay triangles only.
    This is a lower bound on the true minimum (by the theorem, the global min
    must be a Delaunay triangle, but numerical issues may cause slight discrepancies).
    """
    del_tris = _delaunay_triangle_indices(pts)
    if not del_tris:
        return _min_triangle_area(pts)
    
    areas = []
    for (i, j, k) in del_tris:
        pi, pj, pk = pts[i], pts[j], pts[k]
        cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        areas.append(abs(cross) * 0.5)
    
    if not areas:
        return _min_triangle_area(pts)
    return min(areas)


def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_inside_triangle(x, y, eps=1e-9):
    """Check if point (x,y) is inside the unit equilateral triangle."""
    if y < -eps or y > S3H + eps:
        return False
    if x < y / S3 - eps or x > 1 - y / S3 + eps:
        return False
    return True


def _soft_min(areas, beta):
    """
    Soft minimum of areas using log-sum-exp trick.
    This is differentiable and approximates the true minimum as beta -> infinity.
    """
    # We want to minimize the minimum, so we maximize -min
    # soft_min = -log-sum-exp(-beta * areas) / beta
    # = max(areas) - log(sum(exp(-beta*(areas - max(areas))))) / beta
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = max_a - log_sum / beta
    return soft_min_val


def _gradient_descent_delaunay(pts, max_iter=500, lr=0.01, beta=200):
    """
    Gradient descent using Delaunay active set.
    The objective is the soft minimum of Delaunay triangle areas.
    We compute gradients w.r.t. each point based only on its Delaunay neighbors.
    """
    pts = pts.copy()
    best_area = _min_triangle_area(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        # Get Delaunay triangles
        del_tris = _delaunay_triangle_indices(pts)
        if not del_tris:
            # Fallback: use all triangles
            del_tris = _TRIPLETS
        
        # Compute areas of Delaunay triangles
        del_areas = []
        del_tri_list = []
        for (i, j, k) in del_tris:
            pi, pj, pk = pts[i], pts[j], pts[k]
            cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
            area = abs(cross) * 0.5
            del_areas.append(area)
            del_tri_list.append((i, j, k))
        
        del_areas = np.array(del_areas)
        
        # Compute soft minimum gradient
        max_a = np.max(del_areas)
        shifted = -beta * (del_areas - max_a)
        shifted = np.clip(shifted, -500, 0)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp  # softmax weights
        
        # Gradient of soft_min w.r.t. each point
        # soft_min = max_a - log(sum(exp(-beta*(area - max_a)))) / beta
        # d(soft_min)/d(area_m) = -weights[m] * beta * (1/beta) * ... 
        # Actually: soft_min = -log(sum(exp(-beta*area))) / beta
        # d(soft_min)/d(area_m) = -1/beta * (-beta * exp(-beta*area_m) / sum(exp(-beta*area)))
        #                        = exp(-beta*area_m) / sum(exp(-beta*area))
        #                        = weights[m]
        
        # So d(soft_min)/d(point_p) = sum over triangles containing p of weights[tri] * d(area_tri)/d(point_p)
        
        grad = np.zeros_like(pts)
        
        for idx, (i, j, k) in enumerate(del_tri_list):
            w = weights[idx]
            if w < 1e-12:
                continue
            
            pi, pj, pk = pts[i], pts[j], pts[k]
            # d(area)/d(pi) = 0.5 * (pk - pj) rotated by 90 degrees... 
            # area = 0.5 * |cross(pj-pi, pk-pi)|
            # For signed area: A = 0.5 * ((pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0]))
            # dA/d(pi[0]) = 0.5 * (-(pk[1]-pi[1]) + (pj[1]-pi[1])) = 0.5 * (pj[1] - pk[1])
            # dA/d(pi[1]) = 0.5 * ((pk[0]-pi[0]) - (pj[0]-pi[0])) = 0.5 * (pk[0] - pj[0])
            # dA/d(pj[0]) = 0.5 * (pk[1]-pi[1])
            # dA/d(pj[1]) = 0.5 * (-(pk[0]-pi[0])) = 0.5 * (pi[0] - pk[0])
            # dA/d(pk[0]) = 0.5 * (-(pj[1]-pi[1])) = 0.5 * (pi[1] - pj[1])
            # dA/d(pk[1]) = 0.5 * (pj[0]-pi[0])
            
            # But we're using abs, so we need to handle the sign
            cross_val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
            sign = 1 if cross_val >= 0 else -1
            
            dA_dpi = np.array([0.5 * sign * (pj[1] - pk[1]), 0.5 * sign * (pk[0] - pj[0])])
            dA_dpj = np.array([0.5 * sign * (pk[1] - pi[1]), 0.5 * sign * (pi[0] - pk[0])])
            dA_dpk = np.array([0.5 * sign * (pi[1] - pj[1]), 0.5 * sign * (pj[0] - pi[0])])
            
            grad[i] += w * dA_dpi
            grad[j] += w * dA_dpj
            grad[k] += w * dA_dpk
        
        # Update: we want to maximize soft_min, so move in direction of grad
        # But we need to keep points inside the triangle
        for i in range(N):
            pts[i] += lr * grad[i]
        
        # Project to triangle
        pts = _project_to_triangle(pts)
        
        # Check improvement
        current_area = _min_triangle_area(pts)
        if current_area > best_area + 1e-10:
            best_area = current_area
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            # Simple check: if not improving, reduce lr
            pass
    
    return best_pts


def _active_set_optimize(pts, max_iter=300, beta=100):
    """
    Active set optimization: only optimize points that are in the minimum-area
    Delaunay triangles. This is more efficient than full gradient descent.
    """
    pts = pts.copy()
    best_area = _min_triangle_area(pts)
    best_pts = pts.copy()
    lr = 0.02
    
    for it in range(max_iter):
        del_tris = _delaunay_triangle_indices(pts)
        if not del_tris:
            break
        
        # Find the minimum area Delaunay triangle
        min_area = float('inf')
        min_tri_idx = 0
        del_areas = []
        for idx, (i, j, k) in enumerate(del_tris):
            pi, pj, pk = pts[i], pts[j], pts[k]
            cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
            area = abs(cross) * 0.5
            del_areas.append(area)
            if area < min_area:
                min_area = area
                min_tri_idx = idx
        
        del_areas = np.array(del_areas)
        
        # Get points involved in near-minimum triangles
        threshold = min_area + 0.1 * (np.max(del_areas) - min_area + 1e-10)
        active_points = set()
        for idx, (i, j, k) in enumerate(del_tris):
            if del_areas[idx] < threshold:
                active_points.add(i)
                active_points.add(j)
                active_points.add(k)
        
        if not active_points:
            break
        
        # Compute gradient for active points only
        grad = np.zeros_like(pts)
        
        # Use soft min for smoothness
        max_a = np.max(del_areas)
        shifted = -beta * (del_areas - max_a)
        shifted = np.clip(shifted, -500, 0)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        for idx, (i, j, k) in enumerate(del_tris):
            w = weights[idx]
            if w < 1e-10:
                continue
            
            pi, pj, pk = pts[i], pts[j], pts[k]
            cross_val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
            sign = 1 if cross_val >= 0 else -1
            
            dA_dpi = np.array([0.5 * sign * (pj[1] - pk[1]), 0.5 * sign * (pk[0] - pj[0])])
            dA_dpj = np.array([0.5 * sign * (pk[1] - pi[1]), 0.5 * sign * (pi[0] - pk[0])])
            dA_dpk = np.array([0.5 * sign * (pi[1] - pj[1]), 0.5 * sign * (pj[0] - pi[0])])
            
            if i in active_points:
                grad[i] += w * dA_dpi
            if j in active_points:
                grad[j] += w * dA_dpj
            if k in active_points:
                grad[k] += w * dA_dpk
        
        # Update active points
        for i in active_points:
            pts[i] += lr * grad[i]
        
        pts = _project_to_triangle(pts)
        
        current_area = _min_triangle_area(pts)
        if current_area > best_area + 1e-10:
            best_area = current_area
            best_pts = pts.copy()
            lr = min(lr * 1.05, 0.1)  # Increase learning rate slightly
        else:
            lr *= 0.9
            if lr < 1e-4:
                break
    
    return best_pts


def _construct_initial():
    """Create an initial symmetric configuration."""
    # 3 corners
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        # Midpoints of edges
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        # Centroid
        [1/3, S3H/3],
        # Additional points
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    return pts


def _random_init(rng):
    """Generate random points inside the triangle."""
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v <= 1:
                x = u + 0.5 * v
                y = v * S3H
                break
        pts[i] = [x, y]
    return pts


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    best_pts = None
    best_area = 0.0

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Initial configurations ----
    init_sym = _construct_initial()
    _update_best(init_sym)

    # Try a few random inits
    for _ in range(5):
        if not _elapsed_ok(50):
            break
        rand_init = _random_init(rng)
        _update_best(rand_init)

    # ---- Phase 1: Active Set Optimization from best initial ----
    if best_pts is not None and _elapsed_ok(40):
        optimized = _active_set_optimize(best_pts, max_iter=400, beta=150)
        _update_best(optimized)

    # ---- Phase 2: Gradient Descent with Delaunay ----
    if best_pts is not None and _elapsed_ok(35):
        grad_opt = _gradient_descent_delaunay(best_pts, max_iter=300, lr=0.008, beta=200)
        _update_best(grad_opt)

    # ---- Phase 3: Multi-start with active set ----
    for _ in range(8):
        if not _elapsed_ok(15):
            break
        rand_init = _random_init(rng)
        # Quick active set optimization
        opt_rand = _active_set_optimize(rand_init, max_iter=200, beta=100)
        _update_best(opt_rand)

    # ---- Phase 4: Polishing with higher beta ----
    if best_pts is not None and _elapsed_ok(25):
        polished = _active_set_optimize(best_pts, max_iter=500, beta=300)
        _update_best(polished)
        
        polished2 = _gradient_descent_delaunay(best_pts, max_iter=400, lr=0.005, beta=500)
        _update_best(polished2)

    # ---- Phase 5: Final fine-tuning ----
    if best_pts is not None and _elapsed_ok(15):
        # Very small steps
        fine = _active_set_optimize(best_pts, max_iter=800, beta=1000)
        _update_best(fine)

    if best_pts is None:
        return _project_to_triangle(init_sym)

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
