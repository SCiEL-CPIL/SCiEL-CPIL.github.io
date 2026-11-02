"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _in_triangle(p, A, B, C):
    """Check if point p is inside triangle ABC (including boundary)."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, A, B)
    d2 = sign(p, B, C)
    d3 = sign(p, C, A)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def construct():
    """
    Construct 11 points in the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    maximizing the minimum triangle area.
    
    Uses a 3x4 grid topology with rubber band model.
    Optimizes global affine transform + local perturbations.
    """
    sqrt3 = np.sqrt(3.0)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, sqrt3 / 2.0])
    h = sqrt3 / 2.0
    
    # Define 3x4 grid topology: 3 rows, 4 columns
    # Row 0 (bottom, y=0): 4 points
    # Row 1 (middle, y=h/2): 4 points  
    # Row 2 (top, y=h): 3 points (to fit in triangle)
    # Total: 11 points
    
    # Base grid positions in normalized coordinates
    # These will be transformed by affine map
    base_grid = np.array([
        # Row 0 (bottom): 4 points
        [0.0, 0.0], [1/3, 0.0], [2/3, 0.0], [1.0, 0.0],
        # Row 1 (middle): 4 points
        [0.125, 0.5], [0.375, 0.5], [0.625, 0.5], [0.875, 0.5],
        # Row 2 (top): 3 points
        [0.25, 1.0], [0.5, 1.0], [0.75, 1.0],
    ])
    
    # Rubber band connections: connect adjacent points in grid
    # This enforces quasi-lattice structure
    edges = []
    # Horizontal connections within rows
    # Row 0: 0-1, 1-2, 2-3
    edges.extend([(0,1), (1,2), (2,3)])
    # Row 1: 4-5, 5-6, 6-7
    edges.extend([(4,5), (5,6), (6,7)])
    # Row 2: 8-9, 9-10
    edges.extend([(8,9), (9,10)])
    
    # Vertical/diagonal connections between rows
    # Row 0 to Row 1: connect each bottom point to nearest middle points
    edges.extend([(0,4), (0,5), (1,4), (1,5), (1,6), (2,5), (2,6), (2,7), (3,6), (3,7)])
    # Row 1 to Row 2: connect each middle point to nearest top points
    edges.extend([(4,8), (5,8), (5,9), (6,9), (6,10), (7,10)])
    
    # Compute rest lengths from base grid
    rest_lengths = {}
    for i, j in edges:
        rest_lengths[(i,j)] = np.linalg.norm(base_grid[i] - base_grid[j])
    
    def _affine_transform(params, grid):
        """
        Apply affine transform to grid points.
        params: [a, b, c, d, e, f, ...perturbations]
        First 6 params define affine transform:
          x' = a*x + b*y + c
          y' = d*x + e*y + f
        Remaining params are local perturbations for each point (11*2 = 22 values)
        """
        a, b, c, d, e, f = params[:6]
        pts = np.array([
            [a*grid[i,0] + b*grid[i,1] + c, d*grid[i,0] + e*grid[i,1] + f]
            for i in range(11)
        ])
        
        # Add local perturbations
        perturbations = params[6:6+22].reshape(11, 2)
        pts += perturbations
        
        return pts
    
    def _penalty_outside(pts):
        """Penalty for points outside the triangle."""
        pen = 0.0
        for p in pts:
            if not _in_triangle(p, A, B, C):
                # Compute distance to nearest edge for smooth penalty
                # Edge AB: y=0
                if p[1] < 0:
                    pen += 100.0 * (p[1])**2
                # Edge BC: from (1,0) to (0.5, h)
                # Line: sqrt(3)*x + y = sqrt(3)
                dist_bc = (sqrt3 * p[0] + p[1] - sqrt3) / 2.0
                if dist_bc < 0:
                    pen += 100.0 * dist_bc**2
                # Edge CA: from (0.5, h) to (0,0)
                # Line: -sqrt(3)*x + y = 0
                dist_ca = (-sqrt3 * p[0] + p[1]) / 2.0
                if dist_ca < 0:
                    pen += 100.0 * dist_ca**2
        return pen
    
    def _spring_energy(pts):
        """Compute spring potential energy for rubber band model."""
        energy = 0.0
        for i, j in edges:
            d = np.linalg.norm(pts[i] - pts[j])
            rest = rest_lengths[(i,j)]
            energy += 0.5 * (d - rest)**2
        return energy
    
    def objective(params):
        """
        Objective function: minimize -min_area + spring_energy + penalty
        """
        pts = _affine_transform(params, base_grid)
        
        # Penalty for being outside triangle
        pen = _penalty_outside(pts)
        
        # Spring energy
        spring = _spring_energy(pts)
        
        # Min triangle area
        min_area = _min_triangle_area(pts)
        
        # We want to maximize min_area, so minimize -min_area
        # Weight springs lightly to maintain structure
        return -min_area + 0.05 * spring + pen
    
    # Initial parameters
    # Identity affine transform + zero perturbations
    # x' = x, y' = y, so a=1, b=0, c=0, d=0, e=1, f=0
    x0 = np.zeros(6 + 22)
    x0[0] = 1.0  # a
    x0[4] = 1.0  # e
    
    # Try multiple initial configurations
    best_points = None
    best_score = -float('inf')
    
    # Initial: identity transform
    candidates = [x0.copy()]
    
    # Add perturbed initial conditions
    np.random.seed(42)
    for trial in range(10):
        x_trial = x0.copy()
        # Small random perturbations to affine params
        x_trial[0] = 1.0 + 0.1 * np.random.randn()
        x_trial[4] = 1.0 + 0.1 * np.random.randn()
        x_trial[1] = 0.05 * np.random.randn()  # b
        x_trial[3] = 0.05 * np.random.randn()  # d
        x_trial[2] = 0.05 * np.random.randn()  # c
        x_trial[5] = 0.05 * np.random.randn()  # f
        # Small local perturbations
        x_trial[6:] = 0.02 * np.random.randn(22)
        candidates.append(x_trial)
    
    # Try a scaled version that fits better in triangle
    # The base grid spans x:[0,1], y:[0,1] but triangle has y:[0,h]
    # Scale y by h
    x_scaled = x0.copy()
    x_scaled[0] = 1.0
    x_scaled[4] = h  # scale y by h
    candidates.append(x_scaled)
    
    # Try slightly different affine transforms
    for scale_x in [0.9, 1.0, 1.1]:
        for scale_y in [0.9*h, h, 1.1*h]:
            x_try = x0.copy()
            x_try[0] = scale_x
            x_try[4] = scale_y
            candidates.append(x_try)
    
    for x_init in candidates:
        try:
            res = minimize(objective, x_init, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            pts = _affine_transform(res.x, base_grid)
            
            # Check if all points are inside triangle
            valid = True
            for p in pts:
                if not _in_triangle(p, A, B, C):
                    valid = False
                    break
            
            if valid:
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            continue
    
    # If no valid points found, use a fallback configuration
    if best_points is None:
        # Fallback: evenly spaced points in triangle
        best_points = np.array([
            [0.1, 0.05], [0.3, 0.05], [0.5, 0.05], [0.7, 0.05], [0.9, 0.05],
            [0.2, 0.2], [0.4, 0.2], [0.6, 0.2], [0.8, 0.2],
            [0.3, 0.35], [0.5, 0.35], [0.7, 0.35],
        ])[:11]
    
    return np.array(best_points)
# EVOLVE-BLOCK-END
