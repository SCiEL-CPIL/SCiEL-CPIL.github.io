"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE_MARGIN = 140.0

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)

_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _vectorized_min_area_cart(points):
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _project_bary(bary):
    """Project barycentric coords to valid simplex (non-negative, sum=1)."""
    bary = np.maximum(bary, 0.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-15)
    bary = bary / s
    return bary

def _soft_min_bary(bary, alpha):
    """Compute soft-min of triangle areas in barycentric space."""
    cart = _bary_to_cart(bary)
    p1 = cart[_T_I]
    p2 = cart[_T_J]
    p3 = cart[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_a = np.min(areas)
    # Numerically stable log-sum-exp
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-alpha * shifted)))
    soft_min = min_a - (1.0 / alpha) * log_sum
    return soft_min, min_a

def _bary_objective(flat_bary, alpha):
    """Objective in barycentric coordinates: minimize -soft_min (i.e. maximize soft_min)."""
    bary = flat_bary.reshape(N, 3)
    bary = _project_bary(bary)
    soft_min, _ = _soft_min_bary(bary, alpha)
    return -soft_min

def _anneal_optimize(bary0, start_alpha=50.0, end_alpha=500.0, 
                     n_anneal_steps=30, max_nm_iter=200, deadline=None):
    """
    Anneal alpha from low to high, using Nelder-Mead at each step.
    This gradually sharpens the soft-min toward the true min.
    """
    current = bary0.copy()
    current = _project_bary(current)
    
    alphas = np.geomspace(start_alpha, end_alpha, n_anneal_steps)
    
    for i, alpha in enumerate(alphas):
        if deadline is not None and time.time() > deadline:
            break
        
        def obj(flat):
            return _bary_objective(flat, alpha)
        
        res = minimize(obj, current.flatten(), method='Nelder-Mead',
                      options={'maxiter': max_nm_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        
        new_bary = res.x.reshape(N, 3)
        new_bary = _project_bary(new_bary)
        
        # Accept if improved on the actual min area
        _, cur_min = _soft_min_bary(current, 1000.0)
        _, new_min = _soft_min_bary(new_bary, 1000.0)
        
        if new_min >= cur_min:
            current = new_bary
    
    return current

def _symmetric_bary_config():
    """
    Symmetric initial configuration in barycentric coordinates.
    Uses 3-fold rotational symmetry: orbits of size 1 or 3.
    11 = 1 + 3 + 3 + 3 + 1? No. Let's use 1 + 3 + 3 + 4? 
    Better: 1 centroid + 3 + 3 + 3 + 1 = 11? 
    Actually: 1 + 3 + 3 + 3 + 1 doesn't work for symmetry.
    Use: 1 centroid + 3 + 3 + 3 + 1... 
    Let me just place 11 points in a structured pattern.
    """
    bary = np.zeros((N, 3))
    
    # Centroid
    bary[0] = [1/3, 1/3, 1/3]
    
    # 3 points near vertices (orbit)
    r1 = 0.15
    bary[1] = [1 - r1, r1/2, r1/2]  # near vertex 0
    bary[2] = [r1/2, 1 - r1, r1/2]  # near vertex 1
    bary[3] = [r1/2, r1/2, 1 - r1]  # near vertex 2
    
    # 3 points at mid-edges (orbit)
    r2 = 0.35
    bary[4] = [0, r2, 1-r2]  # on edge 1-2
    bary[5] = [r2, 0, 1-r2]  # on edge 0-2
    bary[6] = [1-r2, r2, 0]  # on edge 0-1
    
    # 3 points at inner ring (orbit)
    r3 = 0.25
    bary[7] = [r3, r3, 1-2*r3]
    bary[8] = [r3, 1-2*r3, r3]
    bary[9] = [1-2*r3, r3, r3]
    
    # 1 more point - slightly offset from centroid
    bary[10] = [0.35, 0.30, 0.35]
    
    return bary

def _random_bary():
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _structured_random_bary():
    """Generate a structured random barycentric config."""
    bary = np.zeros((N, 3))
    
    # Centroid-ish
    bary[0] = np.random.dirichlet([1, 1, 1])
    
    # 3 orbits of 3
    for orbit in range(3):
        base = np.random.dirichlet([1, 1, 1])
        # Rotate: permute coordinates
        for k in range(3):
            idx = orbit * 3 + k
            perm = np.roll(base, k)
            bary[idx] = perm
    
    # 1 more
    bary[9] = np.random.dirichlet([1, 1, 1])
    bary[10] = np.random.dirichlet([1, 1, 1])
    
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Start with symmetric configuration
    best_bary = _symmetric_bary_config()
    best_cart = _bary_to_cart(best_bary)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Anneal from symmetric start
    if time.time() < deadline - 15:
        result = _anneal_optimize(best_bary, start_alpha=30.0, end_alpha=400.0,
                                   n_anneal_steps=25, max_nm_iter=150, deadline=deadline - 10)
        result = _project_bary(result)
        cart = _bary_to_cart(result)
        score = _vectorized_min_area_cart(cart)
        if score > best_score:
            best_score = score
            best_bary = result
    
    # Multiple random restarts with annealing
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        # Use structured random for better starts
        if r % 2 == 0:
            bary0 = _structured_random_bary()
        else:
            bary0 = _random_bary()
        
        # Adaptive anneal parameters: fewer steps for later restarts
        steps = max(10, 25 - r // 2)
        result = _anneal_optimize(bary0, start_alpha=20.0, end_alpha=300.0,
                                   n_anneal_steps=steps, max_nm_iter=100, deadline=deadline - 5)
        result = _project_bary(result)
        cart = _bary_to_cart(result)
        score = _vectorized_min_area_cart(cart)
        if score > best_score:
            best_score = score
            best_bary = result
    
    # Final high-precision polish
    if time.time() < deadline - 3:
        result = _anneal_optimize(best_bary, start_alpha=200.0, end_alpha=800.0,
                                   n_anneal_steps=15, max_nm_iter=300, deadline=deadline - 1)
        result = _project_bary(result)
        cart = _bary_to_cart(result)
        score = _vectorized_min_area_cart(cart)
        if score > best_score:
            best_score = score
            best_bary = result
    
    best_cart = _bary_to_cart(_project_bary(best_bary))
    return best_cart
# EVOLVE-BLOCK-END
