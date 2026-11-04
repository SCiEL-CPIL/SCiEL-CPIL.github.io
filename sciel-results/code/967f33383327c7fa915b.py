import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

N = 13

def _calc_score(pts):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns (score, penalty). If penalty > 0, the configuration is invalid.
    """
    # 1. Check convexity / hull vertices
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0, 1e9
        if hull.vertices.size < N:
            # Not all points are on the hull
            # This is a soft constraint in the optimizer, but for the final score
            # we just compute the value.
            # However, for the objective, we want to penalize this to keep them on hull.
            return 0.0, 1e6
    except Exception:
        return 0.0, 1e9

    # 2. Calculate min triangle area
    min_area = float('inf')
    # Vectorized calculation
    # Create indices
    indices = np.indices((N, N, N))
    # Filter i < j < k
    # This is slow in pure python loop, but N=13 is small.
    # Let's use a loop for clarity and speed (286 iterations is nothing)
    
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                areas.append(area)
    
    areas = np.array(areas)
    min_area = np.min(areas)
    
    score = min_area / hull_area
    return score, 0.0

def _objective(x):
    """
    Objective for L-BFGS-B.
    x: array of shape (2*(N-1),)
    Returns -score (to be minimized) + penalty
    """
    # Reconstruct points
    p0 = np.array([1.0, 0.0])
    pts_rest = x.reshape(N-1, 2)
    pts = np.vstack([p0, pts_rest])
    
    # Calculate raw score
    score, penalty = _calc_score(pts)
    
    # We want to maximize score.
    # So minimize -score.
    # Add penalty to keep points on convex hull.
    return -score + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 20s margin
    
    best_score = 0.0
    best_pts = None
    
    # Multi-start
    starts = []
    
    # Start 1: Regular 13-gon
    # Rotate so one point is at (1,0)?
    # Regular polygon vertices: (cos(2*pi*k/N), sin(2*pi*k/N))
    # Point 0 is at (1,0).
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # p0 is (1,0). The rest are pts[1:]
    starts.append(reg_pts[1:].flatten())
    
    # Start 2: Perturbed Regular
    rng = np.random.RandomState(42)
    for _ in range(5):
        noise = rng.normal(0, 0.1, (N-1, 2))
        perturbed = reg_pts[1:] + noise
        starts.append(perturbed.flatten())
        
    # Start 3: Random star-shaped (angles sorted, radii random)
    for _ in range(5):
        # Fix p0 at (1,0)
        # Random angles for others, sorted, in (0, 2pi)
        # Ensure they don't cross p0?
        # p0 is at angle 0.
        # Others should be in (0, 2pi).
        angles = np.sort(rng.uniform(0.1, 2*np.pi - 0.1, N-1))
        radii = rng.uniform(0.5, 1.5, N-1)
        pts_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        starts.append(pts_rand.flatten())
        
    for x0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            # Check result
            if res.success:
                score, penalty = _calc_score(np.vstack([np.array([1.0, 0.0]), res.x.reshape(N-1, 2)]))
                if penalty == 0 and score > best_score:
                    best_score = score
                    best_pts = np.vstack([np.array([1.0, 0.0]), res.x.reshape(N-1, 2)])
                    
        except Exception as e:
            pass
            
    # If no valid points found, return a default (regular polygon)
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts

# EVOLVE-BLOCK-END
