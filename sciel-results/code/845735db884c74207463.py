"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

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
            penalty += (0 - y) * 100
        val_left = np.sqrt(3) * x - y
        if val_left < 0:
            penalty += (-val_left) * 100
        val_right = np.sqrt(3) * (1 - x) - y
        if val_right < 0:
            penalty += (-val_right) * 100
    return penalty

def construct():
    """
    Construct 11 points in the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    maximizing the minimum triangle area.
    Uses a 3x4 grid topology with affine transformation and local perturbations.
    """
    sqrt3 = np.sqrt(3.0)
    h = sqrt3 / 2.0
    
    # Base grid: 11 points in a unit square [0,1]x[0,1]
    # Topology: 4-4-3 arrangement (bottom, middle, top)
    # This creates a "3x4" like structure (3 rows, max 4 cols)
    base_points = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], # Row 0 (4 pts)
        [0.2, 0.5], [0.4, 0.5], [0.6, 0.5], [0.8, 0.5], # Row 1 (4 pts)
        [0.3, 0.9], [0.5, 0.9], [0.7, 0.9]              # Row 2 (3 pts)
    ])
    
    # Affine transform parameters: 6 (a,b,c,d,e,f)
    # P = [[a, b], [c, d]] * G + [e, f]
    # Perturbation parameters: 22 (dx, dy for each of 11 points)
    # Total variables: 6 + 22 = 28
    
    def params_to_points(params):
        # Extract affine params
        a, b, c, d, e, f = params[:6]
        # Extract perturbations
        perturbs = params[6:].reshape(11, 2)
        
        # Apply affine transform
        # G is (11, 2)
        # P_affine = G @ [[a, c], [b, d]] + [e, f]
        # Note: Standard affine is P = M @ G + T
        # M = [[a, b], [c, d]] usually means x' = ax + by + e, y' = cx + dy + f
        # Let's use:
        # x' = a*x + b*y + e
        # y' = c*x + d*y + f
        
        x_aff = base_points[:, 0] * a + base_points[:, 1] * b + e
        y_aff = base_points[:, 0] * c + base_points[:, 1] * d + f
        
        P_affine = np.column_stack((x_aff, y_aff))
        
        # Add perturbations
        P = P_affine + perturbs
        return P

    def objective(params):
        pts = params_to_points(params)
        
        # Rubber band penalty: keep perturbations small
        perturbs = params[6:].reshape(11, 2)
        rubber_penalty = np.sum(perturbs**2) * 0.1
        
        min_area = _min_triangle_area(pts)
        pen_out = _penalty_outside(pts)
        
        # Maximize min_area -> Minimize -min_area
        return -min_area + pen_out + rubber_penalty

    # Initial guess
    # We want to map the unit square roughly to the equilateral triangle.
    # Target triangle vertices: (0,0), (1,0), (0.5, h)
    # Base points are in [0,1]x[0,1].
    # Let's try an affine map that sends (0,0)->(0,0), (1,0)->(1,0), (0,1)->(0.5, h).
    # x' = x
    # y' = y * h + x * 0 ? No.
    # Line from (0,0) to (1,0) is y=0. Base y=0.
    # Line from (0,0) to (0,1) in base -> (0,0) to (0.5, h) in target?
    # If we map (0,1) to (0.5, h), then:
    # x' = a*x + b*y + e. (0,0)->(0,0) => e=0. (1,0)->(1,0) => a=1.
    # (0,1)->(0.5, h) => b=0.5.
    # y' = c*x + d*y + f. (0,0)->(0,0) => f=0. (1,0)->(1,0) => c=0.
    # (0,1)->(0.5, h) => d=h.
    # So:
    # x' = x + 0.5y
    # y' = h*y
    
    # Check corners:
    # (0,0) -> (0,0)
    # (1,0) -> (1,0)
    # (0,1) -> (0.5, h)
    # (1,1) -> (1.5, h) -- Outside!
    # The base square [0,1]x[0,1] mapped this way covers a parallelogram (0,0)-(1,0)-(1.5,h)-(0.5,h).
    # The target triangle is (0,0)-(1,0)-(0.5,h).
    # The base points at x=0.7, 0.8, 0.9 might be outside.
    # Our base points:
    # Max x is 0.8. Max y is 0.9.
    # Point (0.8, 0.9) -> x' = 0.8 + 0.5*0.9 = 1.25. y' = h*0.9.
    # x'=1.25 is outside the triangle (max x at y=h*0.9 is 1 - 0.9 = 0.1? No.
    # Triangle width at y: w = 1 - y/h.
    # At y=0.9h, w = 0.1. Center x=0.5. Range [0.45, 0.55].
    # So x'=1.25 is way outside.
    
    # We need a better initial affine map or better base points.
    # Let's use base points that are already in a triangle shape.
    # Base triangle: (0,0), (1,0), (0.5, 1).
    # Grid in this triangle:
    # Row 0 (y=0): 4 pts. x = 0.1, 0.3, 0.5, 0.7.
    # Row 1 (y=0.5): 4 pts. x = 0.2, 0.4, 0.6, 0.8. (Width at y=0.5 is 0.5. Center 0.5. Range 0.25-0.75. 0.8 is outside.)
    # Let's adjust base points to be strictly inside the unit triangle (0,0)-(1,0)-(0.5,1).
    
    base_points = np.array([
        [0.1, 0.0], [0.3, 0.0], [0.5, 0.0], [0.7, 0.0], # Row 0
        [0.25, 0.5], [0.4, 0.5], [0.6, 0.5], [0.75, 0.5], # Row 1 (Width 0.5, center 0.5. 0.25..0.75 fits)
        [0.35, 0.9], [0.5, 0.9], [0.65, 0.9]              # Row 2 (Width 0.1, center 0.5. 0.45..0.55. 0.35 is outside!)
    ])
    
    # Let's refine base points to be safe.
    # Row 2 y=0.9. Width = 1-0.9=0.1. Center 0.5. Range [0.45, 0.55].
    # 3 points in [0.45, 0.55].
    # x = 0.45, 0.50, 0.55.
    
    base_points = np.array([
        [0.1, 0.0], [0.3, 0.0], [0.5, 0.0], [0.7, 0.0], # Row 0
        [0.25, 0.5], [0.4, 0.5], [0.6, 0.5], [0.75, 0.5], # Row 1
        [0.45, 0.9], [0.5, 0.9], [0.55, 0.9]              # Row 2
    ])
    
    # Now, map this unit triangle (0,0)-(1,0)-(0.5,1) to the target triangle (0,0)-(1,0)-(0.5,h).
    # This is a simple scaling in y.
    # x' = x
    # y' = y * h
    # a=1, b=0, c=0, d=h, e=0, f=0.
    
    initial_affine = [1.0, 0.0, 0.0, h, 0.0, 0.0]
    initial_perturbs = np.zeros((11, 2))
    
    initial_params = np.concatenate([initial_affine, initial_perturbs.flatten()])
    
    # Optimize
    res = minimize(objective, initial_params, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    pts = params_to_points(res.x)
    
    # Final check and return
    if _penalty_outside(pts) < 1e-6:
        return pts
    else:
        # Fallback: return the initial affine mapped points (which are inside)
        return params_to_points(initial_params)

# EVOLVE-BLOCK-END
