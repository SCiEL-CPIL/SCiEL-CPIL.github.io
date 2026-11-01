import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

N = 14

def calc_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
    
    # Calculate all triangle areas
    # Vectorized approach for triangle areas
    # We can compute areas for all triples (i, j, k)
    # To be efficient, we can loop over i, and vectorize j, k?
    # Or just use a simple loop since N=14 is small (364 triangles)
    
    min_area = float('inf')
    
    # Using itertools for clarity, but a numpy broadcast might be faster
    # Let's try a vectorized approach for speed
    # p_i, p_j, p_k
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    
    # Create indices
    # This might be memory heavy if not careful, but 14 is small.
    # Let's just loop, it's safer and fast enough for 364 items.
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Triangle area
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def neg_score(points_flat):
    """
    Objective function for minimization (we want to maximize score).
    points_flat: (2*N,) array
    """
    points = points_flat.reshape(N, 2)
    score = calc_score(points)
    # We want to maximize score, so minimize -score
    # Add a small penalty if points are collinear or degenerate
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    # Initial guesses
    # 1. Regular Polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Elliptical Regular Polygon
    init2 = np.column_stack([1.5*np.cos(angles), np.sin(angles)])
    
    # 3. Random points on a circle with noise
    rng = np.random.RandomState(42)
    angles_rand = np.sort(rng.uniform(0, 2*np.pi, N))
    radii = rng.uniform(0.8, 1.0, N)
    init3 = np.column_stack([radii * np.cos(angles_rand), radii * np.sin(angles_rand)])
    
    inits = [init1, init2, init3]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Normalize init to unit area? Not strictly necessary for optimization,
        # but helps with scale. The score is scale invariant.
        
        x0 = init.flatten()
        
        try:
            # Use Nelder-Mead for non-smooth objective
            res = scipy.optimize.minimize(
                neg_score, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if time.time() > deadline:
                break
                
            current_points = res.x.reshape(N, 2)
            current_score = calc_score(current_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points
                
        except Exception:
            continue
            
    # If best_points is still None, return a regular polygon
    if best_points is None:
        best_points = init1
        
    return best_points
