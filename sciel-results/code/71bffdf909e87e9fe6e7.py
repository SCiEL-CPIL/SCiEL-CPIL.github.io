# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 11

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _penalty_outside(points):
    """
    Calculate penalty for points outside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    sqrt3 = np.sqrt(3.0)
    h = sqrt3 / 2.0
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += (0 - y) * 1000
        val_left = np.sqrt(3) * x - y
        if val_left < 0:
            penalty += (-val_left) * 1000
        val_right = np.sqrt(3) * (1 - x) - y
        if val_right < 0:
            penalty += (-val_right) * 1000
    return penalty

def construct():
    """
    Construct 11 points in the equilateral triangle.
    Uses a symmetric 3-row topology (4-4-3 points) with 8 variables.
    """
    sqrt3 = np.sqrt(3.0)
    h = sqrt3 / 2.0
    
    def params_to_points(params):
        # params: [y0, y1, y2, dx01, dx02, dx11, dx12, dx21]
        y0, y1, y2 = params[0], params[1], params[2]
        dx01, dx02 = params[3], params[4]
        dx11, dx12 = params[5], params[6]
        dx21 = params[7]
        
        pts = []
        # Row 0: 4 points (2 pairs)
        pts.append((0.5 - dx02, y0))
        pts.append((0.5 - dx01, y0))
        pts.append((0.5 + dx01, y0))
        pts.append((0.5 + dx02, y0))
        
        # Row 1: 4 points (2 pairs)
        pts.append((0.5 - dx12, y1))
        pts.append((0.5 - dx11, y1))
        pts.append((0.5 + dx11, y1))
        pts.append((0.5 + dx12, y1))
        
        # Row 2: 3 points (1 pair + 1 center)
        pts.append((0.5 - dx21, y2))
        pts.append((0.5, y2))
        pts.append((0.5 + dx21, y2))
        
        return np.array(pts)

    def objective(params):
        pts = params_to_points(params)
        min_area = _min_triangle_area(pts)
        pen = _penalty_outside(pts)
        return -min_area + pen

    # Initial guesses
    # We want to distribute points roughly evenly.
    # y0 ~ 0.1, y1 ~ 0.4, y2 ~ 0.7
    # dx should be such that points are inside.
    
    candidates = []
    
    # Candidate 1: Symmetric grid-like
    # y0=0.1, width~0.9 -> dx01~0.1, dx02~0.3
    # y1=0.4, width~0.5 -> dx11~0.1, dx12~0.2
    # y2=0.7, width~0.1 -> dx21~0.05
    c1 = np.array([0.1, 0.4, 0.7, 0.1, 0.3, 0.1, 0.2, 0.05])
    candidates.append(c1)
    
    # Candidate 2: Different distribution
    c2 = np.array([0.05, 0.35, 0.65, 0.15, 0.35, 0.1, 0.2, 0.04])
    candidates.append(c2)
    
    # Random candidates
    np.random.seed(42)
    for _ in range(50):
        y0 = np.random.uniform(0.01, 0.2)
        y1 = np.random.uniform(y0 + 0.1, 0.5)
        y2 = np.random.uniform(y1 + 0.1, h - 0.01)
        
        # Max width at y is w = 1 - 2y/sqrt(3)
        # Points must be within 0.5 * w from center 0.5
        w0 = max(0.01, 1 - 2*y0/sqrt3)
        w1 = max(0.01, 1 - 2*y1/sqrt3)
        w2 = max(0.01, 1 - 2*y2/sqrt3)
        
        # dx2 > dx1 > dx3 > 0
        # Ensure dx values are within bounds
        dx02 = np.random.uniform(0.1 * w0, 0.45 * w0)
        dx01 = np.random.uniform(0.05 * w0, dx02 - 0.01)
        
        dx12 = np.random.uniform(0.1 * w1, 0.45 * w1)
        dx11 = np.random.uniform(0.05 * w1, dx12 - 0.01)
        
        dx21 = np.random.uniform(0.05 * w2, 0.45 * w2)
        
        c = np.array([y0, y1, y2, dx01, dx02, dx11, dx12, dx21])
        candidates.append(c)

    best_score = -float('inf')
    best_params = None
    
    for cand in candidates:
        res = minimize(objective, cand, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
        
        pts = params_to_points(res.x)
        if _penalty_outside(pts) < 1e-6:
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_params = res.x

    if best_params is None:
        best_params = candidates[0]
        
    return params_to_points(best_params)
# EVOLVE-BLOCK-END
