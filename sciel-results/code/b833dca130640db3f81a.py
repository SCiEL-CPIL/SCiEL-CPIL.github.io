import numpy as np
from scipy.optimize import differential_evolution

def compute_ratio_fast(points):
    """
    Computes the ratio A_min / A_poly for a given set of points.
    Assumes points are in convex position (ordered by angle).
    Uses Shoelace formula for area.
    """
    n = points.shape[0]
    if n < 3:
        return -np.inf
        
    # Check for invalid points
    if not np.all(np.isfinite(points)):
        return -np.inf

    # Calculate Polygon Area (Shoelace Formula)
    # Assuming points are in CCW order
    x = points[:, 0]
    y = points[:, 1]
    # Shift indices
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area_poly = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    
    if area_poly <= 1e-10:
        return -np.inf

    # Calculate all triangle areas
    # Indices for all combinations of 3 points
    # Using np.triu_indices for efficiency? No, simple loops or indices are fine for n=13
    # Vectorized calculation of all 286 triangles
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    a_min = np.min(areas)
    
    if a_min <= 1e-10:
        return -np.inf
        
    return a_min / area_poly

def heilbronn_convex13() -> np.ndarray:
    n = 13
    # Angles for regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    def unpack(params):
        r = params
        return np.column_stack((r * np.cos(angles), r * np.sin(angles)))
        
    def objective(params):
        points = unpack(params)
        # We return negative ratio for minimization
        ratio = compute_ratio_fast(points)
        if ratio == -np.inf:
            return 1e10 # Large penalty
        return -ratio

    # Bounds for radii
    # The optimal configuration is likely close to unit circle.
    # Allow significant variation to find global optimum.
    bounds = [(0.5, 1.5)] * n
    
    # Use Differential Evolution for global optimization
    # popsize=15, maxiter=1000 should be sufficient within 600s
    # Tolerance can be loose as we just want the best found
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=1000, 
        popsize=15, 
        tol=1e-4,
        init='sobol' # Better initialization
    )
    
    best_params = result.x
    best_points = unpack(best_params)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        # Fallback to regular polygon
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
