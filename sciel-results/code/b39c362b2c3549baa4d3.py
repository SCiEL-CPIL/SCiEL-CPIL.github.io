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
DEADLINE_MARGIN = 145.0

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

def _soft_min_area_cart(points, alpha=100):
    """Calculate soft minimum of triangle areas using log-sum-exp."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min: -1/alpha * log(sum(exp(-alpha * areas)))
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # Clip to triangle boundaries
    # y >= 0
    y = np.maximum(0, y)
    # y <= sqrt(3) * x
    y = np.minimum(y, S3 * x)
    # sqrt(3) * x + y <= sqrt(3)
    y = np.minimum(y, S3 - S3 * x)
    
    # Re-clip x after y adjustments
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    # y >= 0
    y = np.maximum(0, y)
    # y <= sqrt(3) * x
    y = np.minimum(y, S3 * x)
    # sqrt(3) * x + y <= sqrt(3)
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
    # Ensure valid barycentric (non-negative, sum to 1)
    bary = np.maximum(0, bary)
    bary /= bary.sum(axis=1, keepdims=True)
    
    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)
    
    # Convert back to barycentric for consistency
    bary_new = _cart_to_bary(cart)
    
    # Soft minimum area
    soft_min = _soft_min_area_cart(cart, alpha)
    
    # Penalty for points outside triangle (should be small after projection)
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def _cartesian_objective(flat_cart, alpha=100):
    """Objective function in Cartesian coordinates."""
    cart = flat_cart.reshape(N, 2)
    cart = _project_cartesian(cart)
    
    soft_min = _soft_min_area_cart(cart, alpha)
    
    # Penalty for points outside triangle
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Initial best
    best_bary = _random_bary()
    best_cart = _bary_to_cart(best_bary)
    best_cart = _project_cartesian(best_cart)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60:
        def de_objective(flat_bary):
            bary = flat_bary.reshape(N, 3)
            bary = np.maximum(0, bary)
            bary_sum = bary.sum(axis=1, keepdims=True)
            bary = bary / np.maximum(bary_sum, 1e-10)
            cart = _bary_to_cart(bary)
            cart = _project_cartesian(cart)
            # Use a moderate alpha for DE
            soft_min = _soft_min_area_cart(cart, alpha=50)
            x = cart[:, 0]
            y = cart[:, 1]
            pen1 = np.sum(np.maximum(0, -y)**2)
            pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
            pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
            return -soft_min + 100 * (pen1 + pen2 + pen3)
        
        # Bounds for barycentric coordinates [0, 1]
        bounds = [(0, 1)] * (N * 3)
        
        de_res = differential_evolution(de_objective, bounds,
                                        maxiter=100, popsize=20,
                                        tol=1e-4, seed=42,
                                        polish=True, init='latinhypercube')
        
        if de_res.success:
            new_bary = de_res.x.reshape(N, 3)
            new_bary = np.maximum(0, new_bary)
            new_bary /= new_bary.sum(axis=1, keepdims=True)
            new_cart = _bary_to_cart(new_bary)
            new_cart = _project_cartesian(new_cart)
            new_score = _vectorized_min_area_cart(new_cart)
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary
                best_cart = new_cart
    
    # Phase 2: Soft-min annealing with Nelder-Mead
    alphas = [20, 50, 100, 200, 500, 1000, 2000]
    
    for alpha in alphas:
        if time.time() > deadline - 15:
            break
        
        # Number of restarts depends on remaining time
        remaining = deadline - time.time()
        num_restarts = max(3, min(15, int(remaining / 5)))
        
        for r in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Start from current best or random
            if r == 0:
                x0 = best_bary.flatten()
            elif r == 1:
                # Perturbed best
                x0 = best_bary.flatten() + np.random.normal(0, 0.1, N * 3)
                x0 = np.maximum(0, x0)
            else:
                x0 = _random_bary().flatten()
            
            # Optimize with Nelder-Mead in barycentric space
            res = minimize(_bary_objective, x0, method='Nelder-Mead',
                          args=(alpha,),
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            # Evaluate the result
            new_bary = res.x.reshape(N, 3)
            new_bary = np.maximum(0, new_bary)
            bary_sum = new_bary.sum(axis=1, keepdims=True)
            new_bary = new_bary / np.maximum(bary_sum, 1e-10)
            
            new_cart = _bary_to_cart(new_bary)
            new_cart = _project_cartesian(new_cart)
            
            new_score = _vectorized_min_area_cart(new_cart)
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary
                best_cart = new_cart
    
    # Phase 3: Final high-precision polish using auxiliary variable t
    if time.time() < deadline - 5:
        # Use the epigraph formulation: maximize t such that all triangle areas >= t
        # This is equivalent to maximizing the minimum area
        
        def polish_objective(flat_params):
            # flat_params: [x0,y0, x1,y1, ..., x10,y10, t]
            cart = flat_params[:N*2].reshape(N, 2)
            t = flat_params[N*2]
            cart = _project_cartesian(cart)
            
            # Compute all triangle areas
            p1 = cart[_T_I]
            p2 = cart[_T_J]
            p3 = cart[_T_K]
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            # Penalty if any area < t
            violations = np.maximum(0, t - areas)
            penalty = np.sum(violations**2)
            
            # Maximize t - penalty
            return -(t - 0.1 * penalty)
        
        # Initial guess: current best cart and t = current best score
        x0 = np.concatenate([best_cart.flatten(), [best_score]])
        
        # Multiple polish attempts with different initializations
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            
            if attempt == 0:
                init = x0
            else:
                # Slightly perturb
                init = x0.copy()
                init[:N*2] += np.random.normal(0, 0.01, N*2)
                init[N*2] = best_score * (1 + np.random.uniform(-0.05, 0.05))
            
            res = minimize(polish_objective, init, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9})
            
            final_cart = res.x[:N*2].reshape(N, 2)
            final_cart = _project_cartesian(final_cart)
            final_score = _vectorized_min_area_cart(final_cart)
            
            if final_score > best_score:
                best_score = final_score
                best_cart = final_cart
    
    return best_cart
# EVOLVE-BLOCK-END
