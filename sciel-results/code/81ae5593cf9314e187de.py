"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

def is_inside_triangle(p):
    # Barycentric coordinate technique
    # p = b1*V1 + b2*V2 + b3*V3
    # b1 + b2 + b3 = 1
    # Check if all b_i >= -eps
    x, y = p
    # V1=(0,0), V2=(1,0), V3=(0.5, h)
    h = np.sqrt(3)/2
    # b3 = y / h
    b3 = y / h
    # b2 = (x - 0.5*b3) / (1 - 0.5) ? No.
    # x = b2*1 + b3*0.5
    # y = b3*h
    # b1 = 1 - b2 - b3
    b3 = y / h
    b2 = x - 0.5 * b3
    b1 = 1 - b2 - b3
    eps = 1e-9
    return b1 >= -eps and b2 >= -eps and b3 >= -eps

def min_triangle_area(points):
    """
    points: (N, 2) array
    returns: minimum area of any triangle formed by 3 points
    """
    n = len(points)
    min_area = np.inf
    # Vectorized calculation for all combinations is possible but loop is fine for N=11
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    """
    params: flat array of 2*N coordinates
    returns: -min_area (to be minimized)
    """
    points = params.reshape((N, 2))
    
    # Check if all points are inside the triangle
    # Use a small margin to avoid boundary issues during optimization
    for p in points:
        if not is_inside_triangle(p):
            return 1e9  # Large penalty
    
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Initial guess: Hexagonal grid
    # Generate a hexagonal grid of points and scale to fit
    # We want 11 points.
    # A 4x3 grid in hex coordinates has 12 points. Let's try that and remove one?
    # Or just generate a random cluster?
    # Let's use a structured hex grid.
    
    # Hex grid generation
    # Row r, Col c
    # x = c + 0.5*(r%2)
    # y = r * sqrt(3)/2
    
    # We need to select 11 points from a larger hex grid that fits the triangle aspect ratio.
    # Triangle width 1, height ~0.866.
    # Let's try a grid of 5 rows?
    # Row 0: 4 pts
    # Row 1: 4 pts
    # Row 2: 3 pts
    # Total 11.
    
    coords = []
    # Row 0 (bottom)
    for c in range(4):
        coords.append([c, 0])
    # Row 1
    for c in range(4):
        coords.append([c + 0.5, np.sqrt(3)/2])
    # Row 2
    for c in range(3):
        coords.append([c + 0.5, np.sqrt(3)])
        
    # This creates a trapezoid shape. 
    # Let's scale it to fit the triangle.
    # Find bounding box
    coords = np.array(coords)
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    # Scale to fit width 1 and height sqrt(3)/2
    # The triangle is not a rectangle, so we need to be careful.
    # Scale x by 1/(max_x - min_x)
    # Scale y by (sqrt(3)/2) / (max_y - min_y)
    # But this distorts the hexagon.
    
    # Better: Use a uniform scale factor and center it.
    # Width of this grid: 3.5 (from 0 to 3.5)
    # Height: sqrt(3)
    # Triangle width: 1, Height: sqrt(3)/2
    # Aspect ratio of grid: 3.5 / sqrt(3) ~ 2.02
    # Aspect ratio of tri: 1 / (sqrt(3)/2) ~ 1.15
    # The grid is too wide.
    
    # Let's try a different topology.
    # 3 rows:
    # Row 0: 4 pts
    # Row 1: 4 pts
    # Row 2: 3 pts
    # This is what I did.
    
    # Let's just use a random restart with local optimization.
    # It's more robust than guessing the topology.
    
    best_points = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    for restart in range(5):
        if time.time() - start_time > timeout:
            break
            
        # Generate initial points
        if restart == 0:
            # Try the hexagonal structure scaled and rotated/shifted to fit
            # Use a simpler hexagonal cluster
            # Center at (0.5, sqrt(3)/6) - centroid of triangle
            # Generate 11 points in a hexagonal pattern
            # 1 center, 6 ring 1, 4 from ring 2?
            # Let's just use a grid and scale non-uniformly to fit the triangle bounds roughly.
            
            # Grid 4x3
            pts = []
            for r in range(3):
                for c in range(4):
                    x = c + 0.5*(r%2)
                    y = r * np.sqrt(3)/2
                    pts.append([x, y])
            # Remove one point to get 11? Or just use 12 and let optimizer merge?
            # No, N is fixed.
            # Let's use 11 points from a 4x3 grid (12 points) by removing the last one.
            pts = pts[:11]
            pts = np.array(pts)
            
            # Scale to fit triangle
            # Normalize to [0,1] x [0,1] roughly then map to triangle?
            # Simple scale:
            w = pts[:, 0].max() - pts[:, 0].min()
            h = pts[:, 1].max() - pts[:, 1].min()
            
            # Target width ~0.8, height ~0.6 to stay inside
            scale_x = 0.8 / w
            scale_y = 0.6 / h
            scale = min(scale_x, scale_y)
            
            pts[:, 0] *= scale
            pts[:, 1] *= scale
            
            # Center in triangle
            cx = 0.5
            cy = np.sqrt(3)/6
            pts[:, 0] += cx - pts[:, 0].mean()
            pts[:, 1] += cy - pts[:, 1].mean()
            
            # Check if inside, if not, pull in
            for i in range(N):
                if not is_inside_triangle(pts[i]):
                    # Move towards center
                    pts[i] = 0.8 * pts[i] + 0.2 * np.array([0.5, np.sqrt(3)/6])
            
            init_params = pts.flatten()
        else:
            # Random points inside triangle
            pts = []
            for _ in range(N):
                while True:
                    u = rng.random()
                    v = rng.random()
                    if u + v < 1:
                        p = u * V2 + v * V3
                        # Add small margin
                        p = p * 0.9 + 0.1 * np.array([0.5, np.sqrt(3)/6])
                        pts.append(p)
                        break
            init_params = np.array(pts).flatten()
        
        # Optimize
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < best_score:
                best_score = -res.fun
                best_points = res.x.reshape((N, 2))
        except Exception:
            pass
            
    if best_points is None:
        # Fallback to random
        pts = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    p = u * V2 + v * V3
                    pts.append(p)
                    break
        return np.array(pts)
    
    return best_points
# EVOLVE-BLOCK-END
