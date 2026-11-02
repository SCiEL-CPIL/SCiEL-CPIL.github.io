"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use Barycentric coordinates (u, v) where u, v >= 0 and u + v <= 1.
   This naturally represents the interior of the equilateral triangle with vertices
   (0,0), (1,0), (0,1) in barycentric space, which maps to (0,0), (1,0), (1/2, sqrt(3)/2) in Cartesian.
   This eliminates complex geometric constraints.
2. Use a Vectorized Nelder-Mead (Derivative-Free) optimization.
   The objective is the minimum triangle area. This function is non-smooth,
   making gradient-based methods (like SLSQP or L-BFGS-B) prone to getting stuck
   or requiring complex smoothing approximations. Nelder-Mead handles non-smooth
   functions well and is robust in high dimensions (22D for 11 points).
3. Multi-start with diverse initializations to escape local minima.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    Barycentric (u, v) corresponds to:
    P = u * A + v * B + (1 - u - v) * C
    Where A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2) in the standard mapping?
    Wait, standard barycentric for triangle with vertices V0, V1, V2:
    P = l0*V0 + l1*V1 + l2*V2, sum li = 1.
    
    Let's use the mapping:
    V0 = (0,0)
    V1 = (1,0)
    V2 = (0.5, S3H)
    
    If we use coordinates (u, v) such that:
    x = u * 1 + v * 0.5
    y = u * 0 + v * S3H
    Then the third barycentric coordinate is 1 - u - v.
    For the point to be inside the triangle, we need u >= 0, v >= 0, 1-u-v >= 0.
    This matches the domain u, v >= 0, u+v <= 1.
    """
    x = bary[:, 0] + 0.5 * bary[:, 1]
    y = S3H * bary[:, 1]
    return np.column_stack((x, y))

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points (Cartesian)."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_triplets[:, 2]]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _min_triangle_area_bary(bary):
    """Compute min triangle area given barycentric coordinates."""
    cart = _bary_to_cart(bary)
    return _min_triangle_area_cart(cart)

def _objective(x):
    """
    Objective function for minimization.
    We want to MAXIMIZE the minimum area.
    So we MINIMIZE the negative of the minimum area.
    """
    bary = x.reshape(N, 2)
    # Check validity: u >= 0, v >= 0, u+v <= 1
    # Nelder-Mead can step outside. We penalize invalid points heavily.
    if np.any(bary[:, 0] < 0) or np.any(bary[:, 1] < 0) or np.any(bary[:, 0] + bary[:, 1] > 1):
        # Calculate how far out it is to create a smooth penalty
        # Simple large penalty is often sufficient for NM if initial points are good
        # But let's project to boundary for area calculation if slightly out? 
        # No, better to penalize so it moves back in.
        return 1e6
    return -_min_triangle_area_bary(bary)

def _greedy_init_bary(rng):
    """
    Greedy initialization in barycentric space.
    Place points to maximize minimum distance in Cartesian space.
    """
    pts_bary = []
    # Start with a random point
    u, v = rng.random(), rng.random()
    while u + v > 1:
        u, v = rng.random(), rng.random()
    pts_bary.append(np.array([u, v]))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(200):
            u, v = rng.random(), rng.random()
            while u + v > 1:
                u, v = rng.random(), rng.random()
            cand = np.array([u, v])
            
            # Calculate distance to existing points in Cartesian
            cand_cart = _bary_to_cart(cand.reshape(1, 2))[0]
            existing_cart = _bary_to_cart(np.array(pts_bary))
            dists = np.linalg.norm(existing_cart - cand_cart, axis=1)
            min_d = np.min(dists)
            
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts_bary.append(best_pos)
    
    return np.array(pts_bary)

def _structured_init_bary():
    """
    Create a structured initialization in barycentric coordinates.
    Vertices: (0,0), (1,0), (0,1) in barycentric (u,v) space.
    Note: (0,1) in (u,v) maps to x=0.5, y=S3H.
    """
    # A simple grid or known good configuration
    # Let's try a triangular lattice
    pts = []
    # Corners
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.0, 1.0])
    # Midpoints of edges
    pts.append([0.5, 0.0])
    pts.append([0.0, 0.5])
    pts.append([0.5, 0.5]) # Center of triangle
    # Quarter points
    pts.append([0.25, 0.0])
    pts.append([0.75, 0.0])
    pts.append([0.0, 0.25])
    pts.append([0.0, 0.75])
    pts.append([0.25, 0.25])
    return np.array(pts)

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_score = -np.inf
    best_pts_bary = None
    
    # Multi-start initializations
    inits = []
    
    # 1. Greedy
    inits.append(_greedy_init_bary(rng))
    
    # 2. Structured
    inits.append(_structured_init_bary())
    
    # 3. Random starts
    for _ in range(5):
        u = np.random.rand(N)
        v = np.random.rand(N)
        # Rejection sampling for triangle
        mask = u + v > 1
        while np.any(mask):
            u[mask] = np.random.rand(np.sum(mask))
            v[mask] = np.random.rand(np.sum(mask))
            mask = u + v > 1
        inits.append(np.column_stack((u, v)))
    
    # Optimization settings for Nelder-Mead
    # Initial simplex size
    # We need 2N+1 points for the simplex in 2N dimensions.
    # Nelder-Mead handles this internally.
    
    for init_bary in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        x0 = init_bary.flatten()
        
        try:
            # Use Nelder-Mead (Powell is also an option, but NM is standard for this)
            # maxiter needs to be high enough to converge
            # fatol and xatol for convergence
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'fatol': 1e-10,
                    'xatol': 1e-10,
                    'adaptive': True # Helps with scaling issues
                }
            )
            
            # Check if result is valid (inside triangle)
            res_bary = result.x.reshape(N, 2)
            if np.all(res_bary[:, 0] >= 0) and np.all(res_bary[:, 1] >= 0) and np.all(res_bary[:, 0] + res_bary[:, 1] <= 1 + 1e-9):
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_pts_bary = res_bary.copy()
                    
        except Exception:
            pass

    # If optimization failed or didn't find anything better, use best initial
    if best_pts_bary is None:
        for init_bary in inits:
            score = _min_triangle_area_bary(init_bary)
            if score > best_score:
                best_score = score
                best_pts_bary = init_bary.copy()

    # Final polish: A few iterations of local search using a smaller step size
    # or just return the best found.
    # Given the time budget, we might have time for a second pass on the best result.
    if best_pts_bary is not None and time.time() - start_time < DEADLINE - 10:
        x0 = best_pts_bary.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'fatol': 1e-12,
                    'xatol': 1e-12,
                    'adaptive': True
                }
            )
            res_bary = result.x.reshape(N, 2)
            if np.all(res_bary[:, 0] >= 0) and np.all(res_bary[:, 1] >= 0) and np.all(res_bary[:, 0] + res_bary[:, 1] <= 1 + 1e-9):
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_pts_bary = res_bary.copy()
        except Exception:
            pass

    # Convert to Cartesian for return
    if best_pts_bary is None:
        # Fallback: random points
        u = np.random.rand(N)
        v = np.random.rand(N)
        mask = u + v > 1
        while np.any(mask):
            u[mask] = np.random.rand(np.sum(mask))
            v[mask] = np.random.rand(np.sum(mask))
            mask = u + v > 1
        best_pts_bary = np.column_stack((u, v))
        
    return _bary_to_cart(best_pts_bary)

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area_cart(pts)}")
