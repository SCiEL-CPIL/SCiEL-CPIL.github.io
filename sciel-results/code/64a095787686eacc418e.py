"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
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

def _uv_to_bary(uv):
    """
    Map (N, 2) array of u,v in [0,1] to barycentric coordinates (N, 3).
    This mapping ensures points are inside the triangle.
    b0 = (1-u)(1-v)
    b1 = u(1-v)
    b2 = v
    """
    u = uv[:, 0]
    v = uv[:, 1]
    b0 = (1 - u) * (1 - v)
    b1 = u * (1 - v)
    b2 = v
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

def _objective_de(uv_flat):
    """
    Objective function for Differential Evolution.
    Input: flattened array of shape (2*N,)
    Returns: negative of the minimum area (to be minimized).
    """
    uv = uv_flat.reshape(N, 2)
    bary = _uv_to_bary(uv)
    cart = _bary_to_cart(bary)
    min_area = _vectorized_min_area_cart(cart)
    return -min_area

def _local_polish_cart(cart_init, max_iter=5000, tol=1e-12):
    """
    High-precision polish using Nelder-Mead in Cartesian coordinates.
    Nelder-Mead is robust for non-smooth functions.
    """
    def obj_cart(cart):
        # Ensure points are inside triangle? 
        # Nelder-Mead might step outside. We will project back if needed, 
        # but for polishing a good candidate, it usually stays close.
        # To be safe, we can add a large penalty if outside.
        x = cart[:, 0]
        y = cart[:, 1]
        tol = 1e-9
        inside = (y >= -tol) & (y <= S3 * x + tol) & (S3 * x + y <= S3 + tol)
        if not np.all(inside):
            return 1e6 - _vectorized_min_area_cart(cart)
        return -_vectorized_min_area_cart(cart)
    
    try:
        res = minimize(obj_cart, cart_init, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
        final_cart = res.x
        # Final projection to ensure validity
        x = final_cart[:, 0].copy()
        y = final_cart[:, 1].copy()
        y = np.maximum(0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        x = np.maximum(0, x)
        x = np.minimum(1, x)
        y = np.maximum(0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        final_cart = np.column_stack([x, y])
        return final_cart, _vectorized_min_area_cart(final_cart)
    except:
        return cart_init, _vectorized_min_area_cart(cart_init)

def _structured_initial_uv():
    """
    Generate a structured initial UV configuration.
    We want points roughly at vertices, midpoints, and centroid.
    Vertices in Bary: (1,0,0), (0,1,0), (0,0,1)
    UV mapping:
    (1,0,0) -> u=0, v=0
    (0,1,0) -> u=1, v=0
    (0,0,1) -> u=0, v=1
    
    Midpoints:
    (0.5, 0.5, 0) -> b2=0 => v=0. b1=0.5 => u(1)=0.5 => u=0.5. So (0.5, 0)
    (0, 0.5, 0.5) -> b0=0 => (1-u)(1-v)=0. b1=0.5, b2=0.5.
       If v=0.5, (1-u)*0.5=0 => u=1. b1=1*0.5=0.5. So (1, 0.5)
    (0.5, 0, 0.5) -> b1=0 => u(1-v)=0 => u=0. b0=0.5 => (1)(1-v)=0.5 => v=0.5. So (0, 0.5)
    
    Centroid:
    (1/3, 1/3, 1/3) -> b2=1/3 => v=1/3.
    b1=1/3 => u(1-1/3)=1/3 => u(2/3)=1/3 => u=0.5. So (0.5, 1/3)
    """
    uv = np.zeros((N, 2))
    # 3 vertices
    uv[0] = [0, 0]
    uv[1] = [1, 0]
    uv[2] = [0, 1]
    # 3 midpoints
    uv[3] = [0.5, 0]
    uv[4] = [1, 0.5]
    uv[5] = [0, 0.5]
    # Centroid
    uv[6] = [0.5, 1/3]
    # 4 more points: Let's place them symmetrically around the centroid
    # Perturb centroid slightly
    uv[7] = [0.4, 0.3]
    uv[8] = [0.6, 0.3]
    uv[9] = [0.5, 0.4]
    uv[10] = [0.5, 0.2]
    return uv

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # 1. Differential Evolution Global Search
    # Bounds for u and v are [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_uv = _structured_initial_uv().flatten()
    best_score = -_objective_de(best_uv)
    
    # Run DE with time limit
    # We use a wrapper to stop if time is running out, but DE doesn't natively support time limits easily.
    # We will estimate iterations. 
    # Population size 100, maxiter 200. 
    # To be safe, we run DE with a moderate maxiter and check time.
    
    try:
        # Use a slightly smaller population and maxiter to ensure it finishes in time
        # 120s budget. DE is stochastic.
        # Let's try popsize=50, maxiter=100.
        # If time allows, we can do more.
        
        # Check time before starting
        if time.time() < deadline - 30:
            # Run DE
            # seed is used for reproducibility
            res = differential_evolution(
                _objective_de, 
                bounds, 
                seed=42,
                popsize=50, 
                maxiter=100,
                tol=1e-8,
                mutation=(0.3, 1.2),
                recombination=0.7,
                polish=False, # We will polish manually
                init='sobol'
            )
            
            if res.success or res.fun < _objective_de(best_uv):
                new_uv = res.x
                new_score = -res.fun
                if new_score > best_score:
                    best_score = new_score
                    best_uv = new_uv
    except Exception as e:
        pass # Fallback to initial if DE fails

    # Convert best UV to Cartesian
    best_bary = _uv_to_bary(best_uv.reshape(N, 2))
    best_cart = _bary_to_cart(best_bary)
    current_score = _vectorized_min_area_cart(best_cart)
    
    # 2. Local Polish with Nelder-Mead
    # We try to polish the result from DE.
    if time.time() < deadline - 10:
        polished_cart, polish_score = _local_polish_cart(best_cart, max_iter=2000, tol=1e-12)
        if polish_score > current_score:
            current_score = polish_score
            best_cart = polished_cart

    # 3. Additional Polish Attempts from Perturbed Versions
    # If we have time, try polishing slightly perturbed versions to escape shallow local minima
    rng = np.random.RandomState(42)
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        # Perturb the current best in Cartesian space
        perturbed_cart = best_cart + rng.normal(0, 0.01, size=(N, 2))
        
        # Project back to triangle
        x = perturbed_cart[:, 0].copy()
        y = perturbed_cart[:, 1].copy()
        y = np.maximum(0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        x = np.maximum(0, x)
        x = np.minimum(1, x)
        y = np.maximum(0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        perturbed_cart = np.column_stack([x, y])
        
        polished_cart, polish_score = _local_polish_cart(perturbed_cart, max_iter=1000, tol=1e-12)
        if polish_score > current_score:
            current_score = polish_score
            best_cart = polished_cart

    return best_cart
# EVOLVE-BLOCK-END
