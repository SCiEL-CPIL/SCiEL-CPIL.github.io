"""Heilbronn triangles: barycentric soft-min annealing with multi-start and Nelder-Mead refinement."""
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

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)

# Triangle vertices in Cartesian
_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _all_areas_cart(points):
    """Calculate all triangle areas for 165 triplets."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _vectorized_min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    return np.min(_all_areas_cart(points))

def _soft_min_area_cart(points, alpha=50.0):
    """Soft minimum of all triangle areas (smooth approximation)."""
    areas = _all_areas_cart(points)
    min_a = np.min(areas)
    shifted = areas - min_a
    # Use log-sum-exp for numerical stability
    log_sum = np.log(np.sum(np.exp(-alpha * shifted)))
    soft_min = min_a - (1.0 / alpha) * log_sum
    return soft_min

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    for _ in range(5):
        y = np.maximum(0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        x = np.maximum(0, x)
        x = np.minimum(1, x)
    
    return np.column_stack([x, y])

def _is_valid_cart(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and 
            np.all(y <= S3 * x + tol) and 
            np.all(S3 * x + y <= S3 + tol))

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _bary_objective(flat_bary, alpha=50.0):
    """
    Objective function in barycentric coordinates.
    Minimizes negative soft-min area + boundary penalty.
    """
    bary = flat_bary.reshape(N, 3)
    # Normalize to ensure valid barycentric coordinates
    bary = np.maximum(0, bary)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    cart = _bary_to_cart(bary)
    
    # Soft-min area
    sm = _soft_min_area_cart(cart, alpha)
    
    # Boundary penalty (should be zero if barycentric coords are valid)
    x = cart[:, 0]
    y = cart[:, 1]
    pen = (np.sum(np.maximum(0, -y)**2) + 
           np.sum(np.maximum(0, y - S3 * x)**2) + 
           np.sum(np.maximum(0, S3 * x + y - S3)**2))
    
    return -sm + 100 * pen

def _cartesian_objective(flat_cart, alpha=50.0):
    """
    Objective function in Cartesian coordinates.
    Minimizes negative soft-min area + boundary penalty.
    """
    cart = flat_cart.reshape(N, 2)
    cart = _project_cartesian(cart)
    
    sm = _soft_min_area_cart(cart, alpha)
    
    x = cart[:, 0]
    y = cart[:, 1]
    pen = (np.sum(np.maximum(0, -y)**2) + 
           np.sum(np.maximum(0, y - S3 * x)**2) + 
           np.sum(np.maximum(0, S3 * x + y - S3)**2))
    
    return -sm + 100 * pen

def _nelder_mead_refine(cart, alpha=100.0, max_iter=500):
    """
    Refine a configuration using Nelder-Mead on the soft-min objective.
    """
    def obj(flat):
        return _cartesian_objective(flat, alpha)
    
    res = minimize(obj, cart.flatten(), method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    new_cart = res.x.reshape(N, 2)
    new_cart = _project_cartesian(new_cart)
    return new_cart

def _anneal_soft_min(bary, alphas, max_iter_each=200, deadline=None):
    """
    Anneal the alpha parameter in soft-min, using Nelder-Mead at each step.
    Starts with small alpha (smooth) and increases to large alpha (sharp min).
    """
    current_bary = bary.copy()
    current_cart = _bary_to_cart(current_bary)
    
    for alpha in alphas:
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Optimize in Cartesian space using Nelder-Mead
        def obj(flat):
            return _cartesian_objective(flat, alpha)
        
        res = minimize(obj, current_cart.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter_each, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        current_cart = res.x.reshape(N, 2)
        current_cart = _project_cartesian(current_cart)
        current_bary = _cart_to_bary(current_cart)
    
    return current_bary, current_cart

def _lattice_initial():
    """
    Create a structured lattice-based initial configuration.
    Places points on a triangular grid pattern.
    """
    pts = np.zeros((N, 2))
    
    # Use a 3x3 grid-like pattern within the triangle
    # Row 0 (bottom): 3 points
    pts[0] = [0.1, 0.05]
    pts[1] = [0.5, 0.05]
    pts[2] = [0.9, 0.05]
    
    # Row 1 (middle): 3 points
    pts[3] = [0.2, 0.25]
    pts[4] = [0.5, 0.25]
    pts[5] = [0.8, 0.25]
    
    # Row 2 (top): 3 points
    pts[6] = [0.35, 0.45]
    pts[7] = [0.5, 0.45]
    pts[8] = [0.65, 0.45]
    
    # 2 more interior points
    pts[9] = [0.4, 0.15]
    pts[10] = [0.6, 0.15]
    
    return _project_cartesian(pts)

def _symmetric_initial():
    """Create a symmetric initial configuration for 11 points."""
    pts = np.zeros((N, 2))
    
    # Corners (slightly inset)
    pts[0] = [0.05, 0.03]
    pts[1] = [0.95, 0.03]
    pts[2] = [0.5, S3H - 0.05]
    
    # Midpoints of sides (slightly inset)
    pts[3] = [0.5, 0.03]
    pts[4] = [0.72, S3H * 0.5]
    pts[5] = [0.28, S3H * 0.5]
    
    # Interior points - hexagonal pattern
    cx, cy = 0.5, S3H / 3
    r = 0.12
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts[6 + i] = [cx + r * np.cos(angle), cy + r * np.sin(angle) * 0.577]
    
    return _project_cartesian(pts)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Phase 1: Generate multiple diverse initial configurations
    initial_configs = []
    
    # Symmetric
    initial_configs.append(_symmetric_initial())
    
    # Lattice
    initial_configs.append(_lattice_initial())
    
    # Random barycentric starts
    for _ in range(5):
        bary = _random_bary()
        cart = _bary_to_cart(bary)
        initial_configs.append(cart)
    
    # Perturbed symmetric
    for _ in range(3):
        cart = _symmetric_initial() + np.random.randn(N, 2) * 0.08
        initial_configs.append(_project_cartesian(cart))
    
    # Phase 2: Optimize each configuration with soft-min annealing
    best_cart = initial_configs[0]
    best_score = _vectorized_min_area_cart(best_cart)
    
    alphas = [10, 30, 50, 100, 200, 400]
    
    for idx, init_cart in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        # Anneal soft-min
        bary = _cart_to_bary(init_cart)
        bary_out, cart_out = _anneal_soft_min(bary, alphas, max_iter_each=150, deadline=deadline)
        
        score = _vectorized_min_area_cart(cart_out)
        
        if score > best_score:
            best_score = score
            best_cart = cart_out.copy()
    
    # Phase 3: Nelder-Mead refinement with increasing alpha
    for alpha in [100, 200, 500]:
        if time.time() > deadline - 10:
            break
        
        refined = _nelder_mead_refine(best_cart, alpha=alpha, max_iter=300)
        score = _vectorized_min_area_cart(refined)
        
        if score > best_score:
            best_score = score
            best_cart = refined.copy()
    
    # Phase 4: Final fine-tuning with very high alpha
    if time.time() < deadline - 5:
        refined = _nelder_mead_refine(best_cart, alpha=1000, max_iter=500)
        score = _vectorized_min_area_cart(refined)
        if score > best_score:
            best_score = score
            best_cart = refined.copy()
    
    # Ensure final result is valid
    best_cart = _project_cartesian(best_cart)
    
    return best_cart
# EVOLVE-BLOCK-END
