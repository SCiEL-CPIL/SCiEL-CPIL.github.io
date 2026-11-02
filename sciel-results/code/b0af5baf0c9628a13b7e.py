import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

# Precompute indices for all combinations of 3 points
# C(14, 3) = 364
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        # Check if points are collinear or degenerate
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Strategy: Try multiple initializations and refine them.
    # 1. Regular Polygon
    # 2. Grid-like (4x4 with 2 removed/adjusted)
    # 3. Random
    # 4. Halton/Low Discrepancy
    
    initializations = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    initializations.append(x_reg)
    
    # 2. Grid 4x4 (16 points, remove 2 to get 14). 
    # A 4x4 grid in a square has area 9 (if spacing 1).
    # Removing corners or center might help.
    # Let's just use a perturbed 4x4 grid.
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    # Remove 2 points (e.g., last two) to get 14
    x_grid = grid_pts[:14]
    # Scale to roughly unit area or just let optimizer handle scale
    # The score is normalized by hull area, so scale doesn't matter for score, 
    # but it helps optimizer convergence if points are well-spread.
    initializations.append(x_grid)
    
    # 3. Random points in a circle
    rng = np.random.default_rng(123)
    for _ in range(3):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        x_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initializations.append(x_rand)
        
    # 4. Halton sequence (low discrepancy)
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = []
    for i in range(1, N+1):
        x = halton(i, 2)
        y = halton(i, 3)
        halton_pts.append([x, y])
    x_halton = np.array(halton_pts)
    initializations.append(x_halton)
    
    # Refinement function
    def refine(points_init, method='Nelder-Mead', maxiter=2000):
        # Normalize initial points to center and unit scale for better conditioning?
        # Actually, the score is scale invariant (min_area / hull_area).
        # But the optimizer might struggle if scale is huge or tiny.
        # Let's center and scale to unit norm roughly.
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        # Scale to max norm 1
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'Nelder-Mead':
                options = {'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                res = minimize(_objective, pts.flatten(), method='Nelder-Mead', options=options)
            elif method == 'Powell':
                # Powell can be slower but sometimes escapes local minima better
                # or handles flat regions differently.
                res = minimize(_objective, pts.flatten(), method='Powell', 
                               options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6})
            else:
                return None, -np.inf
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = -res.fun
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    # Run refinement on each initialization
    # We have 140 seconds. 6 initializations. ~23s each.
    # Nelder-Mead with 28 variables (14*2) can take a while.
    
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
            
        # Try Nelder-Mead first
        cand_pts, cand_score = refine(init_pts, 'Nelder-Mead', maxiter=1500)
        
        if cand_pts is not None and cand_score > best_score:
            best_score = cand_score
            best_points = cand_pts
            
        # If time allows, try Powell on the best result found so far from this init?
        # Or try Powell on the original init if NM failed?
        # Let's try Powell on the result of NM if it improved, as a second pass.
        if cand_pts is not None and cand_score > best_score and time.time() < deadline - 20.0:
            cand_pts2, cand_score2 = refine(cand_pts, 'Powell', maxiter=500)
            if cand_pts2 is not None and cand_score2 > best_score:
                best_score = cand_score2
                best_points = cand_pts2

    # If we still have time, do a final random perturbation refinement on the global best
    if best_points is not None and time.time() < deadline - 15.0:
        rng2 = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            noise = rng2.normal(0, 0.05, size=(N, 2))
            perturbed = best_points + noise
            cand_pts, cand_score = refine(perturbed, 'Nelder-Mead', maxiter=1000)
            if cand_pts is not None and cand_score > best_score:
                best_score = cand_score
                best_points = cand_pts

    if best_points is None:
        # Fallback to regular polygon
        best_points = initializations[0]
        
    return best_points
