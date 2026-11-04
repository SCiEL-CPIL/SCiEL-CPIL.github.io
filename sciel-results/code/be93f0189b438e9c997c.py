"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

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
    
    # 3. Three Rings: 4, 5, 5
    r1, r2, r3 = 0.3, 0.6, 1.0
    a1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    a3 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    p3 = np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
    three_ring_pts = np.vstack([p1, p2, p3])
    inits.append(three_ring_pts)
    
    # 4. Perturbed Regular Polygon
    rng = np.random.RandomState(123)
    perturbed = inits[0] + rng.normal(0, 0.1, size=(N, 2))
    inits.append(perturbed)
    
    # 5. Another perturbation
    rng2 = np.random.RandomState(456)
    perturbed2 = inits[0] + rng2.normal(0, 0.2, size=(N, 2))
    inits.append(perturbed2)
    
    # 6. Random points in a circle
    rng3 = np.random.RandomState(789)
    rand_angles = rng3.uniform(0, 2 * np.pi, N)
    rand_radii = rng3.uniform(0.5, 1.0, N)
    rand_pts = np.column_stack([rand_radii * np.cos(rand_angles), rand_radii * np.sin(rand_angles)])
    inits.append(rand_pts)

    def objective(x):
        return -_score(x.reshape(-1, 2))
        
    for idx_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        current_score = _score(init_pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = init_pts.copy()
            
        # Optimize with Powell
        try:
            res = minimize(objective, init_pts.flatten(), method='Powell', 
                           options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-12})
            if -res.fun > best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2)
        except Exception:
            pass
            
        # Try Nelder-Mead as well if time permits
        if time.time() < deadline - 15.0:
            try:
                res_nm = minimize(objective, init_pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_pts = res_nm.x.reshape(-1, 2)
            except Exception:
                pass

    # Additional refinement: use the best point found so far and run more iterations
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            res_final = minimize(objective, best_pts.flatten(), method='Powell', 
                               options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-14})
            if -res_final.fun > best_score:
                best_score = -res_final.fun
                best_pts = res_final.x.reshape(-1, 2)
        except Exception:
            pass

    if best_pts is None:
        return inits[0]
        
    return best_pts
# EVOLVE-BLOCK-END
