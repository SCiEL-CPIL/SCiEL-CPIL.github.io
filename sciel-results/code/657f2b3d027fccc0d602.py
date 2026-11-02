"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for degeneracy
    if np.allclose(points, points[0]):
        return 0.0

    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0

    # Compute min triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) |
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    
    if min_area == float('inf'):
        return 0.0
        
    return min_area / hull_area

def _negative_score(flat_points):
    points = flat_points.reshape(-1, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit
    
    best_score = 0.0
    best_points = None
    
    # Initial configurations
    inits = []
    
    # 1. Regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg14 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg14)
    
    # 2. Regular 14-gon with radial perturbation
    rng = np.random.default_rng(42)
    radii = 1.0 + 0.1 * rng.normal(0, 1, N)
    pert14 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    inits.append(pert14)
    
    # 3. Two concentric 7-gons
    # 7 points on r=1, 7 points on r=0.5, interleaved
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    inner_angles = angles7 + np.pi / 7
    inner = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    # Interleave: outer[0], inner[0], outer[1], inner[1]...
    two_ring = np.empty((14, 2))
    for i in range(7):
        two_ring[2*i] = outer[i]
        two_ring[2*i+1] = inner[i]
    inits.append(two_ring)
    
    # 4. Random points in unit circle
    random_points = rng.random((N, 2)) * 2 - 1
    # Ensure they are inside circle
    dists = np.linalg.norm(random_points, axis=1)
    mask = dists > 1.0
    random_points[mask] /= dists[mask, None]
    inits.append(random_points)
    
    # Optimize each initial configuration
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Flatten for scipy
        x0_flat = x0.flatten()
        
        # Use Nelder-Mead for robustness without gradients
        # maxiter is adjusted to fit within time
        try:
            res = minimize(_negative_score, x0_flat, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < -best_score:
                curr_score = -res.fun
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = res.x.reshape(-1, 2)
        except Exception:
            pass
            
        # If we have time, try Powell on the best found so far
        if best_points is not None and time.time() < deadline - 10:
            try:
                x_best_flat = best_points.flatten()
                res2 = minimize(_negative_score, x_best_flat, method='Powell',
                                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10})
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_points = res2.x.reshape(-1, 2)
            except Exception:
                pass

    # If no optimization succeeded or score is 0, return regular 14-gon
    if best_points is None or best_score <= 0:
        best_points = inits[0]
        
    return best_points
# EVOLVE-BLOCK-END
