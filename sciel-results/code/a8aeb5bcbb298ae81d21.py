import numpy as np
from scipy.optimize import differential_evolution
import time

def compute_scores(points):
    """
    Compute the vertex scores for a given set of points.
    points: (20, 2) array.
    Returns: array of 20 scores.
    """
    n = len(points)
    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return np.zeros(n)
    points = points / mean_dist
    
    # Check convexity
    # Compute cross products of consecutive edges
    # Edge i: p[i+1] - p[i]
    # Edge i-1: p[i] - p[i-1]
    # Cross product z-component: (p[i+1]-p[i]) x (p[i]-p[i-1])
    # In 2D: (x1*y2 - y1*x2)
    
    p = points
    edges = np.roll(p, -1, axis=0) - p
    # Cross product of edge i and edge i-1
    # edge_i = p[i+1] - p[i]
    # edge_{i-1} = p[i] - p[i-1]
    # cross = edge_i[0]*edge_{i-1}[1] - edge_i[1]*edge_{i-1}[0]
    # Wait, standard orientation:
    # v1 = p[i] - p[i-1]
    # v2 = p[i+1] - p[i]
    # cross = v1[0]*v2[1] - v1[1]*v2[0]
    # If all cross > 0, it's convex CCW.
    
    v1 = np.roll(p, 1, axis=0) - p
    v2 = np.roll(p, -1, axis=0) - p
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    if np.any(cross <= 0):
        return np.zeros(n)
        
    # Compute pairwise distances
    # dist[i, j] = ||p[i] - p[j]||
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    scores = np.zeros(n)
    for i in range(n):
        d = dists[i]
        d[i] = np.inf # Exclude self
        d_sorted = np.sort(d)
        
        if np.any(d_sorted < 0.05):
            scores[i] = 0
            continue
            
        # Find k minimizing d_{k+1} - d_k
        # d_sorted has 20 elements, last is inf.
        # We care about indices 0..18.
        d_relevant = d_sorted[:19]
        gaps = np.diff(d_relevant)
        k = np.argmin(gaps)
        D = (d_relevant[k] + d_relevant[k+1]) / 2.0
        
        if D < 0.1:
            scores[i] = 0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = d_relevant[0:4]
        elif k == 17:
            d_sel = d_relevant[15:19]
        else:
            d_sel = d_relevant[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        
        score = min(1.0, 1.0 / raw)
        scores[i] = score
        
    return scores

def objective(params):
    """
    Objective function for optimization.
    params: array of 20 radii.
    Returns: -min_score (to be minimized).
    """
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = params
    if np.any(radii <= 0):
        return 1000.0 # Penalty
        
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    scores = compute_scores(points)
    min_score = np.min(scores)
    return -min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Initial guess: regular polygon
    initial_radii = np.ones(n)
    
    # Define bounds for radii
    # Keep them close to 1 to maintain convexity
    # If radii vary too much, convexity might break.
    # For n=20, a variation of +/- 0.2 is likely safe.
    lower = np.full(n, 0.8)
    upper = np.full(n, 1.2)
    
    # Start time
    start_time = time.time()
    timeout = 550.0 # 50s margin
    
    # Differential Evolution
    # We can try a few different strategies or just one robust run.
    # Since the landscape might be complex, DE is a good choice.
    
    try:
        result = differential_evolution(
            objective,
            bounds=list(zip(lower, upper)),
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True, # L-BFGS polish at the end
            init='sobol'
        )
        
        if result.success and result.fun < 0: # -min_score < 0 means min_score > 0
            best_radii = result.x
        else:
            best_radii = initial_radii
            
    except Exception:
        best_radii = initial_radii
        
    # Check if we have time for a local refinement or multiple seeds
    elapsed = time.time() - start_time
    if elapsed < timeout - 10:
        # Try a few more seeds or refine
        # Let's try to refine the best found so far with L-BFGS directly on the coordinates?
        # No, keep the parameterization.
        # Maybe try a different seed.
        pass
        
    # Construct final points
    x = best_radii * np.cos(angles)
    y = best_radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    # Final validation and normalization
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist > 0.01:
        points = points / mean_dist
    else:
        # Fallback to regular polygon
        x = np.cos(angles)
        y = np.sin(angles)
        points = np.column_stack((x, y))
        
    return points
