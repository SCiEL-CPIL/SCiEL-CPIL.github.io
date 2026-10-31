"""Heilbronn triangle problem: 11 points in unit-area equilateral triangle."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11

def _min_area_bary(points_bary):
    """Compute minimum triangle area for points given in barycentric coords (u, v) where w=1-u-v.
    points_bary: (N, 2) array.
    Returns the minimum area (in units of the triangle area).
    """
    n = len(points_bary)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    # Area of triangle formed by 3 points in barycentric coords is:
    # |det([u1,v1,w1],[u2,v2,w2],[u3,v3,w3])| / 2
    # Actually, the area ratio to the full triangle is |det| where det is the 3x3 determinant
    # of the barycentric coordinates.
    # Let's verify: if points are vertices (1,0,0), (0,1,0), (0,0,1), det = 1, area ratio = 1. Correct.
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Determinant of the 3x3 matrix with rows [u,v,w]
                det = (u[i] * (v[j] * w[k] - w[j] * v[k]) 
                     - v[i] * (u[j] * w[k] - w[j] * u[k]) 
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                a = abs(det)
                if a < min_area:
                    min_area = a
    return min_area

def _objective(params):
    """Objective function for optimization: negative of min area."""
    points_bary = params.reshape(N, 2)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    # Check validity: all barycentric coords must be non-negative
    if np.any(u < -1e-10) or np.any(v < -1e-10) or np.any(w < -1e-10):
        return 1e10
    
    # Clamp small negative values to 0
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    w = np.maximum(w, 0)
    
    # Renormalize if needed (should be close to 1)
    s = u + v + w
    if np.any(s < 1e-10):
        return 1e10
    u /= s
    v /= s
    w /= s
    
    points_bary = np.column_stack([u, v])
    min_a = _min_area_bary(points_bary)
    return -min_a

def _bary_to_cart(points_bary):
    """Convert barycentric coords (u, v, w) to Cartesian in the unit equilateral triangle."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    # Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    h = np.sqrt(3) / 2
    x = v * 1.0 + w * 0.5
    y = w * h
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    h = np.sqrt(3) / 2
    
    # Strategy:
    # 1. Use differential evolution for global search
    # 2. Polish with L-BFGS-B or Nelder-Mead
    
    best_points_bary = None
    best_score = -float('inf')
    
    # Bounds for u, v: [0, 1] (w = 1-u-v will be checked)
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Try differential evolution first
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True,
                update_interval=10,
                workers=1
            )
            
            if de_result.success or de_result.fun < best_score:
                score = -de_result.fun
                if score > best_score:
                    best_score = score
                    points_bary = de_result.x.reshape(N, 2)
                    # Clean up
                    u = np.maximum(points_bary[:, 0], 0)
                    v = np.maximum(points_bary[:, 1], 0)
                    w = 1.0 - u - v
                    if np.all(w >= -1e-10):
                        u = np.maximum(u, 0)
                        v = np.maximum(v, 0)
                        w = np.maximum(w, 0)
                        s = u + v + w
                        u /= s
                        v /= s
                        points_bary = np.column_stack([u, v])
                        best_points_bary = points_bary
        except Exception:
            pass
    
    # Try Nelder-Mead from multiple starting points
    rng = np.random.RandomState(123)
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Generate random initial points in barycentric coords
        bary_init = rng.dirichlet([1, 1, 1], size=N)
        u_init = bary_init[:, 0]
        v_init = bary_init[:, 1]
        x0 = np.concatenate([u_init, v_init])
        
        try:
            nm_result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            score = -nm_result.fun
            if score > best_score:
                best_score = score
                points_bary = nm_result.x.reshape(N, 2)
                u = np.maximum(points_bary[:, 0], 0)
                v = np.maximum(points_bary[:, 1], 0)
                w = 1.0 - u - v
                if np.all(w >= -1e-10):
                    u = np.maximum(u, 0)
                    v = np.maximum(v, 0)
                    w = np.maximum(w, 0)
                    s = u + v + w
                    u /= s
                    v /= s
                    points_bary = np.column_stack([u, v])
                    best_points_bary = points_bary
        except Exception:
            pass
    
    # If no good result, use a simple grid
    if best_points_bary is None:
        # Create a triangular grid of 11 points
        # Rows: 1, 2, 3, 2, 1 = 9 points, need 2 more
        # Let's use a 4x4 grid subset
        grid_points = []
        for i in range(4):
            for j in range(4 - i):
                # Barycentric coordinates for grid points
                # Level i from bottom
                u = j / 3.0
                v = i / 3.0
                w = 1 - u - v
                if w >= -1e-10:
                    grid_points.append([u, v])
        
        # We have 1+2+3+4 = 10 points, need 1 more
        # Add center point
        grid_points.append([1/3, 1/3])
        
        best_points_bary = np.array(grid_points)
    
    # Convert to Cartesian
    points_cart = _bary_to_cart(best_points_bary)
    
    # Final validation: ensure points are inside the triangle
    # Clamp to triangle
    for i in range(N):
        x, y = points_cart[i]
        eps = 1e-9
        # y >= 0
        y = max(eps, y)
        # y <= sqrt(3)*x
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x - eps
        # y <= sqrt(3)*(1-x)
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x) - eps
        # x >= 0, x <= 1
        x = max(eps, min(1 - eps, x))
        points_cart[i] = [x, y]
    
    return points_cart
