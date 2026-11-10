# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the minimum triangle area
    to the convex hull area.
    """
    n = 13
    start_time = time.time()
    timeout = 580  # Safety margin for 600s limit
    
    # Precompute indices for all triples
    triples = list(itertools.combinations(range(n), 3))
    
    def get_min_triangle_area(points):
        """Calculate the minimum area of any triangle formed by the points."""
        min_area = float('inf')
        for i, j, k in triples:
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            if area < min_area:
                min_area = area
        return min_area

    def get_hull_area(points):
        """Calculate the area of the convex hull."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            # If hull fails (e.g., collinear), return a large penalty or 0
            # Returning 0 would make the ratio undefined/inf. 
            # We want to avoid this. A small positive number makes the ratio huge (bad for min, good for max?).
            # Actually, if points are collinear, min area is 0, so score is 0.
            # If hull area is 0, we should return a value that indicates badness.
            # Let's return a small epsilon to avoid division by zero, but min area will be ~0 anyway.
            return 1e-10

    def objective(x):
        """Objective function to minimize: negative of the score."""
        points = x.reshape((n, 2))
        
        # Check for validity
        if not np.all(np.isfinite(points)):
            return 1e10
            
        min_area = get_min_triangle_area(points)
        hull_area = get_hull_area(points)
        
        if hull_area < 1e-9:
            return 1e10
            
        score = min_area / hull_area
        return -score

    best_points = None
    best_score = -1.0

    # Strategy: Multiple restarts with different initializations
    # 1. Regular polygon
    # 2. Perturbed regular polygon
    # 3. Random points on circle
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        if r == 0:
            # Regular 13-gon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            x0 = np.column_stack((np.cos(angles), np.sin(angles)))
        elif r == 1:
            # Perturbed regular 13-gon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radii = 1.0 + 0.1 * np.random.normal(0, 0.1, n)
            x0 = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        else:
            # Random points on the unit circle (or slightly inside)
            angles = np.sort(np.random.uniform(0, 2 * np.pi, n))
            radii = np.random.uniform(0.8, 1.0, n)
            x0 = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Flatten x0 for optimizer
        x0_flat = x0.flatten()
        
        try:
            # Use Nelder-Mead for derivative-free optimization
            # maxiter is set high, but we rely on the time check
            result = minimize(objective, x0_flat, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.success or result.fun < objective(x0_flat):
                points = result.x.reshape((n, 2))
                # Verify points are finite
                if np.all(np.isfinite(points)):
                    score = -result.fun
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        
        except Exception:
            continue

    # If no optimization succeeded, fall back to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    # Final check and return
    return best_points

# EVOLVE-BLOCK-END
