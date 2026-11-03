"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
    b2 = y / S3
    b1 = x - 0.5 * b2
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

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
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

def _structured_initial_bary():
    """Generate structured initial barycentric configuration with 6-fold symmetry."""
    # 3 vertices
    v0 = np.array([1, 0, 0])
    v1 = np.array([0, 1, 0])
    v2 = np.array([0, 0, 1])
    # 3 edge midpoints
    m01 = np.array([0.5, 0.5, 0])
    m12 = np.array([0, 0.5, 0.5])
    m02 = np.array([0.5, 0, 0.5])
    # Centroid
    c = np.array([1/3, 1/3, 1/3])
    # 4 more symmetric points: 1/4 and 3/4 along edges
    q1 = np.array([0.75, 0.25, 0])
    q2 = np.array([0.25, 0.75, 0])
    q3 = np.array([0, 0.75, 0.25])
    q4 = np.array([0, 0.25, 0.75])
    
    pts = np.array([v0, v1, v2, m01, m12, m02, c, q1, q2, q3, q4])
    return pts

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _hard_objective_bary(flat):
    """Objective in barycentric coords: negative hard min area."""
    bary = flat.reshape(N, 3)
    bary = np.maximum(bary, 0)
    bary /= bary.sum(axis=1, keepdims=True)
    cart = _bary_to_cart(bary)
    return -_vectorized_min_area_cart(cart)

def _local_polish_powell(bary_init, max_iter=1000, tol=1e-12):
    """High-precision polish using Powell method in barycentric coordinates."""
    bary_flat = bary_init.flatten()
    bounds = [(0.0, 1.0)] * (N * 3)
    
    try:
        res = minimize(_hard_objective_bary, bary_flat, method='Powell',
                      bounds=bounds, options={'maxiter': max_iter, 'ftol': tol, 'xtol': tol})
        final_bary = res.x.reshape(N, 3)
        final_bary = np.maximum(final_bary, 0)
        final_bary /= final_bary.sum(axis=1, keepdims=True)
        final_cart = _bary_to_cart(final_bary)
        return final_cart, _vectorized_min_area_cart(final_cart)
    except:
        cart = _bary_to_cart(bary_init)
        return cart, _vectorized_min_area_cart(cart)

def _basin_hopping_search(start_bary, start_time, deadline, rng):
    """
    Structural change: Basin Hopping search using Powell local optimizer.
    This replaces the soft-min annealing with a direct search on the hard objective,
    using random kicks to escape local minima.
    """
    best_bary = start_bary.copy()
    best_cart = _bary_to_cart(best_bary)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Initial local optimization
    cart, score = _local_polish_powell(best_bary, max_iter=500)
    if score > best_score:
        best_score = score
        best_cart = cart
        best_bary = _cart_to_bary(best_cart)
        
    # Basin Hopping parameters
    T_start = 0.01
    T_end = 1e-6
    n_iterations = 200
    kick_scale = 0.1
    
    for i in range(n_iterations):
        if time.time() > deadline - 10:
            break
            
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (i / n_iterations)
        
        # Generate a kick
        kick = rng.normal(0, kick_scale, size=(N, 3))
        kicked_bary = best_bary + kick
        kicked_bary = np.maximum(kicked_bary, 0)
        kicked_bary /= kicked_bary.sum(axis=1, keepdims=True)
        
        # Local optimization from kicked state
        kicked_cart, kicked_score = _local_polish_powell(kicked_bary, max_iter=200)
        
        # Metropolis acceptance criterion
        delta = kicked_score - best_score
        if delta > 0 or (T > 0 and np.log(rng.uniform()) < delta / T):
            best_bary = _cart_to_bary(kicked_cart)
            best_cart = kicked_cart
            best_score = kicked_score
            
        # Occasionally reset kick scale to larger values to explore more
        if i % 20 == 0:
             kick_scale = 0.1 + 0.1 * (i / n_iterations)

    return best_cart, best_score, best_bary

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    # Start with structured initial configuration
    init_bary = _structured_initial_bary()
    best_cart = _bary_to_cart(init_bary)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Primary search: Basin Hopping with Powell
    if time.time() < deadline - 30:
        cart, score, bary = _basin_hopping_search(init_bary, start_time, deadline, rng)
        if score > best_score:
            best_score = score
            best_cart = cart
            best_bary = bary
    else:
        best_bary = init_bary

    # Secondary search: Random restarts with Powell
    # Use remaining time for a few more independent runs
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        num_restarts = int(remaining_time / 10) # Rough estimate
        num_restarts = max(1, min(num_restarts, 5))
        
        for _ in range(num_restarts):
            if time.time() > deadline - 5:
                break
            random_bary = _random_bary()
            cart, score, bary = _basin_hopping_search(random_bary, start_time, deadline, rng)
            if score > best_score:
                best_score = score
                best_cart = cart
                best_bary = bary

    # Final polish
    if time.time() < deadline - 2:
        polished, polish_score = _local_polish_powell(best_bary, max_iter=1000, tol=1e-14)
        if polish_score > best_score:
            best_score = polish_score
            best_cart = polished

    return best_cart
# EVOLVE-BLOCK-END
