"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

# Safety margin from 150s limit
DEADLINE_MARGIN = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    # b2 = y / S3
    b2 = y / S3
    # b1 = x - 0.5*b2
    b1 = x - 0.5 * b2
    # b0 = 1 - b1 - b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _vectorized_min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_areas_cart(points):
    """Calculate all triangle areas formed by the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_area_cart(points, alpha=100):
    """Calculate soft minimum of triangle areas using log-sum-exp."""
    areas = _all_areas_cart(points)
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # Clip to triangle boundaries
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    
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

def _bary_objective(flat_bary, alpha=100):
    """Objective function in barycentric coordinates."""
    bary = flat_bary.reshape(N, 3)
    bary = np.maximum(0, bary)
    bary /= bary.sum(axis=1, keepdims=True)
    
    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)
    
    soft_min = _soft_min_area_cart(cart, alpha)
    
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def _delaunay_active_set_optimize(points, max_iters=500, tol=1e-10):
    """
    Delaunay-based active set optimization.
    Focus optimization on the critical triangles (smallest areas).
    """
    from scipy.spatial import Delaunay
    
    current = points.copy()
    current = _project_cartesian(current)
    
    for it in range(max_iters):
        areas = _all_areas_cart(current)
        min_a = np.min(areas)
        
        # Find critical triangles (bottom 10% of areas)
        threshold = np.percentile(areas, 10)
        critical_mask = areas <= threshold
        critical_indices = np.where(critical_mask)[0]
        
        # Build objective based only on critical triangles
        def obj(flat):
            c = flat.reshape(N, 2)
            c = _project_cartesian(c)
            
            # Compute areas for critical triangles only
            p1 = c[_T_I[critical_mask]]
            p2 = c[_T_J[critical_mask]]
            p3 = c[_T_K[critical_mask]]
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            crit_areas = 0.5 * np.abs(cross)
            
            min_crit = np.min(crit_areas)
            # Soft min over critical triangles
            alpha = max(50.0, min(500.0, 5.0 / max(min_crit, 1e-6)))
            soft_min = min_crit - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (crit_areas - min_crit))))
            
            # Also include a small penalty for overall min area to prevent drift
            all_areas = _all_areas_cart(c)
            overall_min = np.min(all_areas)
            
            x = c[:, 0]
            y = c[:, 1]
            pen = (np.sum(np.maximum(0, -y)**2) + 
                   np.sum(np.maximum(0, y - S3*x)**2) + 
                   np.sum(np.maximum(0, S3*x + y - S3)**2))
            
            return -soft_min - 0.1 * overall_min + 50.0 * pen
        
        res = minimize(obj, current.flatten(), method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9})
        
        new_point = res.x.reshape(N, 2)
        new_point = _project_cartesian(new_point)
        
        new_score = _vectorized_min_area_cart(new_point)
        if new_score > min_a + tol:
            current = new_point
        else:
            break
    
    return current

def _symmetric_initial_config():
    """Create a symmetric initial configuration."""
    cart = np.zeros((N, 2))
    
    # Bottom row: 4 points
    cart[0] = [0.05, 0.0]
    cart[1] = [0.35, 0.0]
    cart[2] = [0.65, 0.0]
    cart[3] = [0.95, 0.0]
    
    # Middle row: 4 points
    cart[4] = [0.2, S3H * 0.33]
    cart[5] = [0.4, S3H * 0.33]
    cart[6] = [0.6, S3H * 0.33]
    cart[7] = [0.8, S3H * 0.33]
    
    # Top row: 3 points
    cart[8] = [0.3, S3H * 0.67]
    cart[9] = [0.5, S3H * 0.67]
    cart[10] = [0.7, S3H * 0.67]
    
    return cart

def _slp_optimize(x0, max_iters=200, tol=1e-10):
    """SLP optimization with adaptive alpha."""
    current = x0.copy()
    current_obj = _slp_objective(current.flatten())
    
    for it in range(max_iters):
        cart = _project_cartesian(current)
        min_a = _vectorized_min_area_cart(cart)
        
        alpha = max(20.0, min(500.0, 10.0 / max(min_a, 1e-6)))
        
        def obj(flat):
            c = _project_cartesian(flat.reshape(N, 2))
            p1 = c[_T_I]
            p2 = c[_T_J]
            p3 = c[_T_K]
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            ma = np.min(areas)
            sm = ma - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - ma))))
            x = c[:, 0]
            y = c[:, 1]
            pen = (np.sum(np.maximum(0, -y)**2) + 
                   np.sum(np.maximum(0, y - S3*x)**2) + 
                   np.sum(np.maximum(0, S3*x + y - S3)**2))
            return -sm + 50.0 * pen
        
        res = minimize(obj, current.flatten(), method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9})
        
        new_point = res.x.reshape(N, 2)
        new_point = _project_cartesian(new_point)
        new_obj = _slp_objective(new_point.flatten())
        
        if new_obj < current_obj - tol:
            current = new_point
            current_obj = new_obj
        else:
            break
    
    return current

def _slp_objective(flat_cart):
    """SLP style objective with adaptive alpha."""
    cart = flat_cart.reshape(N, 2)
    cart = _project_cartesian(cart)
    
    areas = _all_areas_cart(cart)
    min_a = np.min(areas)
    
    alpha = max(20.0, min(500.0, 10.0 / max(min_a, 1e-6)))
    
    soft_min = min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))
    
    x = cart[:, 0]
    y = cart[:, 1]
    pen = (np.sum(np.maximum(0, -y)**2) + 
           np.sum(np.maximum(0, y - S3 * x)**2) + 
           np.sum(np.maximum(0, S3 * x + y - S3)**2))
    
    return -soft_min + 50.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Initial symmetric configuration
    best_cart = _symmetric_initial_config()
    best_cart = _project_cartesian(best_cart)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Phase 1: SLP optimization from symmetric start
    if time.time() < deadline - 30:
        result = _slp_optimize(best_cart, max_iters=200)
        result = _project_cartesian(result)
        score = _vectorized_min_area_cart(result)
        if score > best_score:
            best_score = score
            best_cart = result
    
    # Phase 2: Delaunay active set optimization
    if time.time() < deadline - 20:
        result = _delaunay_active_set_optimize(best_cart, max_iters=300)
        result = _project_cartesian(result)
        score = _vectorized_min_area_cart(result)
        if score > best_score:
            best_score = score
            best_cart = result
    
    # Phase 3: Random restarts with SLP
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        bary = _random_bary()
        cart = _bary_to_cart(bary)
        cart = _project_cartesian(cart)
        
        result = _slp_optimize(cart, max_iters=100)
        result = _project_cartesian(result)
        score = _vectorized_min_area_cart(result)
        if score > best_score:
            best_score = score
            best_cart = result
    
    # Phase 4: Final polish with Delaunay active set
    if time.time() < deadline - 5:
        result = _delaunay_active_set_optimize(best_cart, max_iters=400)
        result = _project_cartesian(result)
        score = _vectorized_min_area_cart(result)
        if score > best_score:
            best_score = score
            best_cart = result
    
    # Final refinement with high-precision SLP
    if time.time() < deadline - 2:
        result = _slp_optimize(best_cart, max_iters=500, tol=1e-12)
        result = _project_cartesian(result)
        score = _vectorized_min_area_cart(result)
        if score > best_score:
            best_score = score
            best_cart = result
    
    best_cart = _project_cartesian(best_cart)
    return best_cart
# EVOLVE-BLOCK-END
