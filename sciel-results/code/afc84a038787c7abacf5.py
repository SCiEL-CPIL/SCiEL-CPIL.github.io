import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Generate Regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Center at (0,0), radius 1
    pts_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Calculate initial score
    def get_score(pts):
        try:
            # Calculate all triangle areas
            areas = []
            for i in range(N):
                for j in range(i+1, N):
                    for k in range(j+1, N):
                        p1 = pts[i]
                        p2 = pts[j]
                        p3 = pts[k]
                        area = 0.5 * np.abs(np.cross(p2-p1, p3-p1))
                        areas.append(area)
            min_area = np.min(areas)
            
            # Calculate Convex Hull Area
            hull = ConvexHull(pts)
            hull_area = hull.volume # In 2D, .volume is area
            
            if hull_area == 0:
                return 0.0
            return min_area / hull_area
        except Exception:
            return 0.0

    best_score = get_score(pts_reg)
    best_pts = pts_reg.copy()
    
    # 2. Refine with Optimization
    # We fix p0 and p1 to remove translation and rotation invariance.
    # However, the regular polygon is symmetric.
    # Let's just optimize all 28 coordinates but penalize large movements or rely on the symmetry.
    # Actually, fixing p0=(0,0) and p1=(1,0) is standard.
    # We must initialize the other points consistently with the regular polygon.
    
    # Rotate/Scale regular polygon so p0 is (0,0) and p1 is (1,0)?
    # In regular polygon, distance between adjacent vertices is $2 \sin(\pi/N)$.
    # Let's construct the regular polygon such that p0=(0,0) and p1 is at distance $d$ along x-axis.
    d = 2 * np.sin(np.pi / N)
    
    # Construct points
    pts_init = np.zeros((N, 2))
    pts_init[0] = [0, 0]
    pts_init[1] = [d, 0]
    
    # The rest of the points can be calculated geometrically or just taken from the regular polygon
    # and transformed.
    # Easier: Take the regular polygon, find the index of the "closest" to (0,0) and (d,0) in a transformed space?
    # Actually, just take the regular polygon, compute its hull area, and normalize?
    # No, the optimizer will handle the scale if we use the ratio objective.
    
    # Let's use the regular polygon points directly as initial guess for the optimizer.
    # To use the optimizer with fixed p0, p1, we need to transform the regular polygon.
    # Let's just optimize all 28 points. The symmetry will likely be preserved or broken optimally.
    
    def objective(x):
        if time.time() > deadline:
            return 0 # Stop optimization
        pts = x.reshape((N, 2))
        # Normalize to prevent scale drift? 
        # The ratio is scale invariant. The optimizer might drift to huge or small scales.
        # To keep it stable, we can normalize the points to have unit hull area before calculating min area?
        # Or just calculate the ratio.
        
        # Calculate min area
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    area = 0.5 * np.abs(np.cross(p2-p1, p3-p1))
                    areas.append(area)
        min_area = np.min(areas)
        
        # Calculate Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 1.0 # Large value for minimization (bad score)
            
        if hull_area < 1e-10:
            return 1.0
            
        return - (min_area / hull_area) # Negate for minimization

    # Initial guess: Regular polygon
    x0 = pts_reg.flatten()
    
    # Optimize
    # Use Powell method for robustness
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 50, 'maxfev': 1000})
        
        if res.success or res.fun < -best_score: # res.fun is negative score
            score_candidate = -res.fun
            if score_candidate > best_score:
                best_score = score_candidate
                best_pts = res.x.reshape((N, 2))
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 5:
        return best_pts
        
    # Optional: Second pass with tighter tolerance or different method?
    # Given the time limit, one pass is usually sufficient if the initial guess is good.
    
    return best_pts
