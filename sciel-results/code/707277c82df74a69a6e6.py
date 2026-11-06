import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    # We want to maximize S = (prod_{i<j} d_ij) / D^28
    # where D = max_{i<j} d_ij.
    # This is equivalent to maximizing F = sum_{i<j} ln(d_ij) - 28 * ln(D).
    
    n = 8
    
    def objective(params):
        # params is a flat array of 2*n floats
        points = params.reshape((n, 2))
        
        # Calculate all pairwise distances
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < 1e-10:
                    return 1e10 # Avoid log(0)
                dists.append(d)
        
        dists = np.array(dists)
        D = np.max(dists)
        if D < 1e-10:
            return 1e10
            
        # Log product - 28 * log diameter
        val = np.sum(np.log(dists)) - 28 * np.log(D)
        
        # We want to minimize negative val
        return -val

    # Initial guesses
    # 1. Regular Octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # 2. Two squares (one rotated 45 deg)
    # Square 1: radius 1, angles 0, 90, 180, 270
    # Square 2: radius r, angles 45, 135, 225, 315
    # We can try r=1 (regular octagon) but perturb it.
    # Let's try a specific non-regular shape known to be competitive.
    # A "squashed" octagon.
    
    # 3. Random
    np.random.seed(42)
    p_rand = np.random.randn(n * 2)
    
    # 4. Square with 4 points near center
    # Vertices of square at (1,1), (-1,1), (-1,-1), (1,-1)
    # Other 4 at (0, 0.5), (0, -0.5), (0.5, 0), (-0.5, 0) ?
    p_sq = np.array([
        [1, 1], [-1, 1], [-1, -1], [1, -1],
        [0, 0.5], [0, -0.5], [0.5, 0], [-0.5, 0]
    ]).flatten()
    
    starts = [p_oct, p_rand, p_sq]
    
    best_val = -np.inf
    best_points = None
    
    for start in starts:
        try:
            res = minimize(objective, start, method='Powell', options={'maxiter': 500, 'ftol': 1e-8, 'xtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_points = res.x
        except Exception:
            continue
            
    # If optimization failed or didn't improve, fallback to regular octagon
    if best_points is None:
        return [[cos(2 * np.pi * k / 8), sin(2 * np.pi * k / 8)] for k in range(8)]
        
    points = best_points.reshape((n, 2))
    
    # The problem asks for points. The score calculation in the evaluator scales them to diameter 2.
    # We can return the optimized points as is, or scale them to diameter 2 for cleanliness.
    # The evaluator does: "scaling the set to diameter 2".
    # So we just need to return the shape.
    
    # Ensure distinct points
    # Check for duplicates
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-5:
                valid = False
    if not valid:
        return [[cos(2 * np.pi * k / 8), sin(2 * np.pi * k / 8)] for k in range(8)]
        
    return points.tolist()

# Helper for fallback
from math import cos, sin, pi
