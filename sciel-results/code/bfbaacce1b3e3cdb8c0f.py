"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar

# Pre-compute indices for N=14 for fast triangle area calculation
_IDX_CACHE = {}
def _get_indices(n):
    if n not in _IDX_CACHE:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _IDX_CACHE[n] = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    return _IDX_CACHE[n]

def _min_triangle_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_area(pts, beta):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Stable log-sum-exp for soft min
    # min(x) approx -1/beta * log(sum(exp(-beta * x)))
    # To avoid overflow, subtract min
    min_a = np.min(areas)
    # If min_a is very large or small, adjust
    # We want -1/beta * log(sum(exp(-beta * (areas - min_a)))) + min_a
    # exp(-beta * (areas - min_a)) is <= 1, so safe
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - (1.0 / beta) * log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two Rings: 7 inner, 7 outer
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_angles = inner_angles + (2 * np.pi / 14)
    
    inner_pts = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    
    two_ring_pts = np.vstack([inner_pts, outer_pts])
    inits.append(two_ring_pts)
    
    # 3. Perturbed Regular Polygon (small noise)
    rng = np.random.RandomState(123)
    perturbed = inits[0] + rng.normal(0, 0.05, size=(N, 2))
    inits.append(perturbed)
    
    # 4. Perturbed Regular Polygon (medium noise)
    rng2 = np.random.RandomState(456)
    perturbed2 = inits[0] + rng2.normal(0, 0.15, size=(N, 2))
    inits.append(perturbed2)
    
    # 5. Random points in a circle
    rng3 = np.random.RandomState(789)
    rand_angles = rng3.uniform(0, 2 * np.pi, N)
    rand_radii = rng3.uniform(0.5, 1.0, N)
    rand_pts = np.column_stack([rand_radii * np.cos(rand_angles), rand_radii * np.sin(rand_angles)])
    inits.append(rand_pts)

    def true_objective(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)

    def soft_objective(x, beta):
        pts = x.reshape(-1, 2)
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-9:
            return 1e10
        soft_min = _soft_min_area(pts, beta)
        # We want to maximize soft_min / hull_area
        # So minimize - (soft_min / hull_area)
        return -(soft_min / hull_area)

    def optimize_init(init_pts):
        current_score = _score(init_pts)
        best_local_score = current_score
        best_local_pts = init_pts.copy()
        
        # Continuation method for soft-min
        betas = [5.0, 20.0, 100.0, 500.0]
        current_pts = init_pts
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            
            try:
                res = minimize(
                    soft_objective, 
                    current_pts.flatten(), 
                    args=(beta,),
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success or res.status == 0:
                    new_pts = res.x.reshape(-1, 2)
                    new_score = _score(new_pts)
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_local_pts = new_pts.copy()
                    current_pts = new_pts
            except Exception:
                pass
        
        # Final polish with Powell on the true objective
        if time.time() < deadline - 5.0:
            try:
                res_powell = minimize(
                    true_objective, 
                    best_local_pts.flatten(), 
                    method='Powell', 
                    options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10}
                )
                if -res_powell.fun > best_local_score:
                    best_local_score = -res_powell.fun
                    best_local_pts = res_powell.x.reshape(-1, 2)
            except Exception:
                pass

        return best_local_score, best_local_pts

    for idx_init, init_pts in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
            
        local_score, local_pts = optimize_init(init_pts)
        
        if local_score > best_score:
            best_score = local_score
            best_pts = local_pts.copy()

    if best_pts is None:
        return inits[0]
        
    return best_pts
# EVOLVE-BLOCK-END
