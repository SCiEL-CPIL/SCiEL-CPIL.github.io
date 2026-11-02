import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Objective function: maximize min triangle area / hull area
    # We minimize the negative of this value.
    
    def score(pts):
        # pts is (N, 2)
        # Calculate all triangle areas
        # Vectorized calculation of cross products for all triples
        # Area = 0.5 * | (Pj - Pi) x (Pk - Pi) |
        
        # To be efficient, we can compute pairwise differences, but with N=13, 
        # even a loop over 286 triangles is fast.
        # Let's use a vectorized approach for speed.
        
        # P[i, j, :] is point j
        # We want area of (i, j, k)
        
        # Using the formula: Area = 0.5 * | x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j) |
        
        # Get all combinations of 3 indices
        # There are 286 combinations.
        
        # For speed, we can precompute indices
        # But inside the optimization loop, we need to be very fast.
        
        # Let's try a semi-vectorized approach.
        # Create a matrix of all points
        # This might be overkill. A simple loop in Python for 286 items is ~0.1ms.
        
        min_area = float('inf')
        
        # Calculate convex hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # In 2D, volume is area
            if hull_area < 1e-10:
                return 1e10  # Degenerate
        except Exception:
            return 1e10  # Degenerate or error
        
        # Calculate min triangle area
        # Using itertools for clarity, should be fast enough for N=13
        # 286 triangles * 13 points * 2 coords = very small
        
        for i, j, k in itertools.combinations(range(N), 3):
            p1 = pts[i]
            p2 = pts[j]
            p3 = pts[k]
            
            # Cross product magnitude
            area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
            
            if area < min_area:
                min_area = area
        
        if min_area == float('inf'):
             return 1e10
        
        ratio = min_area / hull_area
        return -ratio

    # Initial guess: Regular 13-gon
    # Center at (0,0), radius 1
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # To help the optimizer, we can fix the center of mass to (0,0) and rotation to be arbitrary.
    # Nelder-Mead works well on 26 dimensions.
    
    best_score = -score(x0)
    best_pts = x0.copy()
    
    # Run optimization
    # Nelder-Mead is robust for this scale
    try:
        res = minimize(score, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 5000})
        
        if res.success or res.fun < score(x0):
            # Check if it's better
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_pts = res.x
    except Exception as e:
        pass
    
    # Try a few more restarts with small perturbations if time allows
    rng = np.random.default_rng(42)
    
    for i in range(10):
        if time.time() > deadline:
            break
            
        # Perturb the best solution found so far
        noise = rng.normal(0, 0.1, size=x0.shape)
        x_init = best_pts + noise
        
        # Ensure it's still roughly convex? The objective function handles non-convex by calculating hull area.
        # If points become non-convex (some inside), the hull area shrinks, but min triangle area might also drop.
        # The optimizer should naturally keep them on the hull if that's optimal.
        
        try:
            res = minimize(score, x_init, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': 2000})
            
            if res.fun < score(best_pts): # i.e. -ratio is smaller, so ratio is larger
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_pts = res.x
        except Exception:
            pass
            
    # Normalize the final points?
    # The task says "normalized to unit area".
    # "The notebook normalizes by the point set convex hull area."
    # The score is min_area / hull_area.
    # This ratio is scale-invariant.
    # However, the output should be the points.
    # Usually, it's good to return points in a standard position, e.g., centered at origin.
    
    # Center the points
    centroid = np.mean(best_pts, axis=0)
    best_pts = best_pts - centroid
    
    # Optional: Scale to unit area?
    # The score is invariant to scaling. The evaluator calculates the score from the returned points.
    # So we just need to return valid points.
    # Returning them centered is fine.
    
    return best_pts
