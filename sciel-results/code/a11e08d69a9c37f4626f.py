import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

def construct():
    # 1. Generate a 15-point triangular lattice
    k = 5
    L = 1.0
    h = np.sqrt(3) / 2
    
    points_15 = []
    for j in range(k):
        for i in range(k - j):
            x = (i + 0.5 * j) / (k - 1) * L
            y = j * h / (k - 1) * L
            points_15.append([x, y])
    
    points_15 = np.array(points_15)
    
    # 2. Define the triangle vertices for clipping
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, h])
    
    def min_area(points):
        # Calculate the minimum area of any triangle formed by 3 points
        # points is (N, 2)
        min_a = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k_idx in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k_idx]
                    # Area = 0.5 * |cross(p2-p1, p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_a:
                        min_a = area
        return min_a

    def penalty(points):
        # Penalty for points outside the triangle
        v0 = v3 - v1
        v1v = v2 - v1
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1v)
        d11 = np.dot(v1v, v1v)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return 0
        
        pen = 0.0
        for p in points:
            v2v = p - v1
            d02 = np.dot(v0, v2v)
            d12 = np.dot(v1v, v2v)
            u = (d11 * d02 - d01 * d12) / denom
            v = (d00 * d12 - d01 * d02) / denom
            if u < 0:
                pen += u**2  # Quadratic penalty is smoother
            if v < 0:
                pen += v**2
            if u + v > 1:
                pen += (u + v - 1)**2
        return pen

    def objective(params):
        pts = params.reshape(N, 2)
        p_val = penalty(pts)
        # Use a large multiplier to ensure feasibility
        return -min_area(pts) + 1000 * p_val

    # 3. Evaluate all subsets of 11 points from the 15
    subset_scores = []
    
    indices_list = list(range(15))
    # We want to choose 11 points. It's equivalent to choosing 4 points to REMOVE.
    # C(15, 4) = 1365.
    for remove_indices in itertools.combinations(indices_list, 4):
        keep_indices = [i for i in indices_list if i not in remove_indices]
        subset_points = points_15[keep_indices]
        score = min_area(subset_points)
        subset_scores.append((score, keep_indices))
        
    # Sort by score descending
    subset_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Local Optimization on the top few subsets
    best_score = -float('inf')
    best_points = None
    
    # Try the top 5 subsets to escape local minima
    for score, keep_indices in subset_scores[:5]:
        x0 = points_15[keep_indices].flatten()
        
        # Run Nelder-Mead
        res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 5000, 'maxfev': 5000})
        
        final_points = res.x.reshape(N, 2)
        
        # Check if the final points are valid (inside triangle)
        # The penalty should handle this, but let's verify the score
        # We want to maximize min_area, so we check the actual min_area of the result
        # If penalty was triggered, min_area might be low or the points outside.
        # We should only accept if penalty is small.
        pen_val = penalty(final_points)
        if pen_val < 1e-4:
            current_score = min_area(final_points)
            if current_score > best_score:
                best_score = current_score
                best_points = final_points
        else:
            # If penalty is high, the points might be outside. 
            # We could try to project them back, but that's complex.
            # Given the quadratic penalty, they should be close.
            # If this happens often, we might need a better constraint handling.
            # For now, we skip if penalty is too high.
            pass

    # Fallback: if no valid point found (unlikely), return the best lattice subset
    if best_points is None:
        best_points = points_15[subset_scores[0][1]]

    return best_points

# EVOLVE-BLOCK-END
