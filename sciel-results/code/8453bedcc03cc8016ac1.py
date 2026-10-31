"""Heilbronn triangles: place 11 points to maximize minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
_DEADLINE = time.time() + 135  # Safety margin before 150s kill

def _bary_to_cart(a, b, c):
    """Convert barycentric coords to cartesian in equilateral triangle."""
    return np.array([b + 0.5 * c, (3 ** 0.5) / 2 * c])

def _cart_to_bary(x, y):
    """Convert cartesian to barycentric coords."""
    c = 2 * y / (3 ** 0.5)
    b = x - 0.5 * c
    a = 1 - b - c
    return np.array([a, b, c])

def _min_det_area(bary_points):
    """Compute minimum |det| over all triples of barycentric points."""
    n = len(bary_points)
    min_val = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                mat = np.array([bary_points[i], bary_points[j], bary_points[k]])
                det = abs(np.linalg.det(mat))
                if det < min_val:
                    min_val = det
    return min_val

def _objective(bary_flat):
    """Objective: negative of min det area (to minimize)."""
    bary = bary_flat.reshape(N, 3)
    # Normalize to ensure a+b+c=1
    bary = bary / bary.sum(axis=1, keepdims=True)
    # Clip small negatives
    bary = np.clip(bary, 0, 1)
    return -_min_det_area(bary)

def _generate_initial_grid():
    """Generate initial barycentric points from a structured pattern."""
    # Use a pattern based on dividing the triangle into smaller triangles
    # For 11 points, use a mix of vertices, edge midpoints, and center
    points = []
    
    # Vertices
    points.append(np.array([1.0, 0.0, 0.0]))
    points.append(np.array([0.0, 1.0, 0.0]))
    points.append(np.array([0.0, 0.0, 1.0]))
    
    # Edge midpoints
    points.append(np.array([0.5, 0.5, 0.0]))
    points.append(np.array([0.5, 0.0, 0.5]))
    points.append(np.array([0.0, 0.5, 0.5]))
    
    # Center
    points.append(np.array([1/3, 1/3, 1/3]))
    
    # Additional points - use a symmetric pattern
    # Points at 1/4 and 3/4 along edges
    points.append(np.array([0.25, 0.75, 0.0]))
    points.append(np.array([0.25, 0.0, 0.75]))
    points.append(np.array([0.0, 0.25, 0.75]))
    
    return np.array(points)

def _local_optimize(bary_init, max_iters=5000, time_limit=10):
    """Local optimization using Nelder-Mead."""
    start_time = time.time()
    bary = bary_init.copy()
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    best_val = _objective(bary.flatten())
    best_bary = bary.copy()
    
    result = minimize(
        _objective,
        bary.flatten(),
        method='Nelder-Mead',
        options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    if time.time() - start_time > time_limit:
        return best_bary, best_val
    
    if result.fun < best_val:
        best_val = result.fun
        best_bary = result.x.reshape(N, 3)
        best_bary = best_bary / best_bary.sum(axis=1, keepdims=True)
    
    return best_bary, best_val

def construct():
    start_time = time.time()
    
    # Generate initial configuration
    bary_init = _generate_initial_grid()
    
    # Local optimization
    best_bary, best_score = _local_optimize(bary_init, max_iters=3000, time_limit=30)
    
    # Try multiple random restarts if time allows
    rng = np.random.RandomState(42)
    for restart in range(10):
        if time.time() - start_time > 120:
            break
        
        # Generate random initial points
        bary_rand = rng.dirichlet(np.ones(N) * 2, size=N)
        # Add some structure: mix with grid
        bary_rand = 0.5 * bary_rand + 0.5 * bary_init
        bary_rand = bary_rand / bary_rand.sum(axis=1, keepdims=True)
        
        bary_new, score_new = _local_optimize(bary_rand, max_iters=2000, time_limit=8)
        
        if score_new < best_score:
            best_score = score_new
            best_bary = bary_new
    
    # Convert best barycentric to cartesian
    cart_points = np.array([_bary_to_cart(*b) for b in best_bary])
    
    # Verify points are inside the triangle
    for i in range(N):
        x, y = cart_points[i]
        if x < -1e-9 or y < -1e-9 or x > 1 + 1e-9 or y > (3 ** 0.5) / 2 + 1e-9:
            # Project back into triangle
            bary = _cart_to_bary(x, y)
            bary = np.clip(bary, 0, 1)
            bary = bary / bary.sum()
            cart_points[i] = _bary_to_cart(*bary)
    
    return cart_points
