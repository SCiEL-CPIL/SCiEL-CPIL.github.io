import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time

N = 50

def _calculate_covering_radius(points):
    """
    Calculate the covering radius of the configuration.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
    except Exception:
        return np.pi
        
    max_radius = 0.0
    
    for face in faces:
        v1, v2, v3 = points[face]
        
        # Calculate circumcenter
        u = v2 - v1
        w = v3 - v1
        n = np.cross(u, w)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        # Planar circumcenter
        A = np.array([[np.dot(u, u), np.dot(w, u)], [np.dot(u, w), np.dot(w, w)]])
        B = np.array([np.dot(u, u), np.dot(w, w)])
        
        try:
            coeffs = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            continue
        a, b = coeffs
        c_p = v1 + a * u + b * w
        
        c_p_norm = np.linalg.norm(c_p)
        if c_p_norm < 1e-10:
            continue
            
        c_s = c_p / c_p_norm
        cos_R = np.dot(c_s, v1)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        if R > max_radius:
            max_radius = R
            
    return max_radius

def _objective(flat_points):
    """
    Objective function for the optimizer.
    Takes a flat array of 150 floats, reshapes to (50,3), normalizes,
    and returns the covering radius.
    """
    points = flat_points.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    # Optional: add penalty if points are too close to each other (coincident)
    # But ConvexHull will fail or produce degenerate faces, returning pi.
    
    return _calculate_covering_radius(points)

def construct():
    start_time = time.time()
    max_time = 115.0
    
    # Bounds for differential evolution: [-1, 1] for each coordinate
    bounds = [(-1.0, 1.0)] * (N * 3)
    
    # We can use a seed for reproducibility
    np.random.seed(42)
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            polish=True  # Use L-BFGS-B for final polish, which might be slow but helps
        )
        
        best_points = result.x.reshape(N, 3)
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
        
    except Exception:
        # Fallback to Fibonacci sphere if DE fails or times out
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_points
