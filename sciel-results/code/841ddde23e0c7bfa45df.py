# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0
                areas[i,j,k] = area
                areas[k,j,i] = area
                areas[j,k,i] = area
    return areas

def _soft_min_triangle_area(points, beta=0.01):
    """Compute soft-min of all triangle areas using log-sum-exp."""
    n = len(points)
    # Vectorized triangle area computation
    idx = np.arange(n)
    ii, jj, kk = np.meshgrid(idx, idx, idx, indexing='ij')
    ii = ii[ii < jj]
    jj = jj[ii < jj]
    kk = kk[ii < jj]
    # Filter for i < j < k
    mask = (ii < jj) & (jj < kk)
    ii = ii[mask]
    jj = jj[mask]
    kk = kk[mask]
    
    x1, y1 = points[ii]
    x2, y2 = points[jj]
    x3, y3 = points[kk]
    areas = np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0
    
    # Soft min: -logsumexp(-areas/beta) * beta
    # For numerical stability
    a = -areas / beta
    a_max = np.max(a)
    log_sum = a_max + np.log(np.sum(np.exp(a - a_max)))
    soft_min = -log_sum * beta
    return soft_min

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _objective(x, beta=0.01):
    """Objective to minimize: -soft_min_triangle_area / hull_area"""
    n = len(x) // 2
    points = x.reshape(n, 2)
    
    # Check for degeneracy
    if not np.all(np.isfinite(points)):
        return 1e10
    
    ha = _hull_area(points)
    if ha < 1e-10:
        return 1e10
    
    sma = _soft_min_triangle_area(points, beta)
    if sma < 0:
        return 1e10
    
    return -sma / ha

def _hexagonal_seed():
    """Create a hexagonal lattice seed for 13 points."""
    # Hexagonal lattice points
    # Use a triangular arrangement
    points = []
    
    # Center
    points.append([0.5, 0.5])
    
    # Inner ring: 6 points at radius 0.2
    r1 = 0.2
    for i in range(6):
        angle = i * np.pi / 3
        points.append([0.5 + r1*np.cos(angle), 0.5 + r1*np.sin(angle)])
    
    # Outer ring: 6 points at radius 0.4
    r2 = 0.4
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        points.append([0.5 + r2*np.cos(angle), 0.5 + r2*np.sin(angle)])
    
    return np.array(points)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    # Start with hexagonal seed
    points0 = _hexagonal_seed()
    x0 = points0.flatten()
    
    # Track best
    best_points = points0.copy()
    best_score = -_objective(x0, beta=0.05)
    
    # Multi-stage optimization with decreasing beta
    betas = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
        
        result = minimize(
            _objective,
            current_x,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < -best_score:
            current_x = result.x
            score = -result.fun
            if score > best_score:
                best_score = score
                best_points = current_x.reshape(13, 2)
    
    # Final polish with very small beta
    if time.time() < deadline - 20:
        result = minimize(
            _objective,
            current_x,
            args=(0.0005,),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        score = -result.fun
        if score > best_score:
            best_points = result.x.reshape(13, 2)
            best_score = score
    
    # Verify and return
    final_points = best_points.copy()
    
    # Ensure finite
    if not np.all(np.isfinite(final_points)):
        return _hexagonal_seed()
    
    # Verify convex hull works
    try:
        hull = ConvexHull(final_points)
    except Exception:
        return _hexagonal_seed()
    
    return final_points

# EVOLVE-BLOCK-END
