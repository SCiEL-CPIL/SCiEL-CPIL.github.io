import numpy as np
from scipy.optimize import minimize
import time

N = 11
S3 = np.sqrt(3)
H = S3 / 2.0

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute indices for all triples
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """
    points: (N, 2) array
    Returns minimum triangle area formed by any 3 points.
    """
    idx = _get_indices()
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    
    # Vectors from i to j and i to k
    v1 = points[j] - points[i]
    v2 = points[k] - points[i]
    
    # Cross product (2D) magnitude is |v1_x * v2_y - v1_y * v2_x|
    # Area = 0.5 * |cross|
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _bary_to_cart(bary):
    """
    bary: (N, 3) array of barycentric coordinates (u, v, w)
    Returns (N, 2) array of Cartesian coordinates.
    """
    # P = u*V0 + v*V1 + w*V2
    # V0 is 0, so P = v*V1 + w*V2
    # V1 = [1, 0], V2 = [0.5, H]
    # P_x = v + 0.5*w
    # P_y = H*w
    return np.column_stack((bary[:, 1] + 0.5 * bary[:, 2], H * bary[:, 2]))

def _cart_to_bary(cart):
    """
    cart: (N, 2) array
    Returns (N, 3) array of barycentric coordinates.
    """
    x, y = cart[:, 0], cart[:, 1]
    # y = H * w  => w = y / H
    w = y / H
    # x = v + 0.5 * w => v = x - 0.5 * w
    v = x - 0.5 * w
    # u = 1 - v - w
    u = 1.0 - v - w
    return np.column_stack((u, v, w))

def _clamp_bary(bary):
    """
    Clamp barycentric coordinates to be non-negative and sum to 1.
    """
    bary = np.maximum(bary, 0.0)
    sums = np.sum(bary, axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1.0
    return bary / sums

def _objective_bary(flat_bary):
    """
    Objective function for barycentric coordinates.
    Minimizes negative minimum triangle area.
    """
    bary = flat_bary.reshape(N, 3)
    bary = _clamp_bary(bary)
    cart = _bary_to_cart(bary)
    min_area = _min_triangle_area_vectorized(cart)
    return -min_area

def _make_structured_starts():
    """
    Generate multiple structured initial guesses in barycentric space.
    """
    starts = []
    # 1. Grid-like distribution
    # Try to distribute points roughly evenly
    # A simple heuristic: place points on a small grid in barycentric coords
    # and then optimize.
    
    # Start with a few random restarts and some structured ones
    rng = np.random.default_rng(42)
    
    # Structured: 3x3 grid in barycentric (u,v) with w=1-u-v
    # u, v in [0, 1], u+v <= 1
    # We can sample u, v from a grid
    u_vals = np.linspace(0.1, 0.8, 3)
    v_vals = np.linspace(0.1, 0.8, 3)
    
    # This is too coarse for 11 points. Let's just use random restarts with better seeding
    # and a few specific patterns.
    
    # Pattern 1: Random uniform in barycentric
    for i in range(5):
        bary = rng.dirichlet([1, 1, 1], size=N)
        starts.append(bary.flatten())
        
    # Pattern 2: Clustered near vertices? No, we want spread.
    # Pattern 3: Regular polygon-like in barycentric?
    
    # Let's add a "grid" start
    # Divide triangle into 9 smaller triangles (3x3)
    # Place one point in each? We have 11 points.
    # Let's just use random restarts but ensure we have enough of them.
    
    return starts

def _local_optimize_nm(flat_bary):
    """
    Optimize barycentric coordinates using Nelder-Mead.
    """
    res = minimize(
        _objective_bary,
        flat_bary,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    return res

def _refine_with_multiple_methods(flat_bary, time_limit=10.0):
    """
    Try different local optimization methods or restarts if time permits.
    """
    best_res = _local_optimize_nm(flat_bary)
    best_fun = best_res.fun
    best_x = best_res.x
    
    # If time allows, try a second NM run with different initial simplex?
    # Or try L-BFGS-B? Barycentric coords are non-negative, so L-BFGS-B could work with bounds.
    # But the objective is non-smooth (min of linear functions), so L-BFGS might struggle.
    # NM is generally better for this.
    
    # Maybe try a "jittered" restart?
    # Add small noise to best_x and run NM again
    if time.time() < time_limit:
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.01, size=N*3)
        perturbed = best_x + noise
        res2 = _local_optimize_nm(perturbed)
        if res2.fun < best_fun:
            best_fun = res2.fun
            best_x = res2.x
            
    return best_x, best_fun

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate initial starts
    starts = _make_structured_starts()
    
    # Add a few more random starts
    rng = np.random.default_rng(123)
    for i in range(10):
        bary = rng.dirichlet([1, 1, 1], size=N)
        starts.append(bary.flatten())
        
    best_points = None
    best_score = -float('inf')
    
    for i, start_bary in enumerate(starts):
        if time.time() > deadline:
            break
            
        # Refine this start
        try:
            # Use a short time limit for each refinement to ensure we try many
            # If we have 15 starts and 110s, we have ~7s per start.
            # NM is fast. Let's just run it.
            res_x, res_fun = _refine_with_multiple_methods(start_bary, time_limit=deadline)
            
            if res_fun < best_score:
                best_score = res_fun
                # Convert best barycentric to Cartesian
                bary = res_x.reshape(N, 3)
                bary = _clamp_bary(bary)
                best_points = _bary_to_cart(bary)
                
        except Exception as e:
            continue
            
    if best_points is None:
        # Fallback: random points
        rng = np.random.default_rng(0)
        bary = rng.dirichlet([1, 1, 1], size=N)
        best_points = _bary_to_cart(bary)
        
    return best_points
