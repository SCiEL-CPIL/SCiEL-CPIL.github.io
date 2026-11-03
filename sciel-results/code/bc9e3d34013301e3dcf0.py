import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Triangle vertices
V = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2]
])

def min_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    min_a = float('inf')
    # 11 points -> 165 triangles
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_a:
                    min_a = area
    return min_a

def is_inside_triangle(p, vertices):
    """
    Check if point p is inside the triangle defined by vertices.
    Uses barycentric coordinate technique.
    """
    x, y = p
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    
    # Calculate the area of the triangle
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if area == 0:
        return False

    # Calculate the areas of the three sub-triangles
    a1 = 0.5 * abs((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y))
    a2 = 0.5 * abs((x2 - x) * (y3 - y) - (x3 - x) * (y2 - y))
    a3 = 0.5 * abs((x3 - x) * (y1 - y) - (x1 - x) * (y3 - y))
    
    # If the point is inside, the sum of sub-areas equals the total area
    # We use a small tolerance
    return abs(a1 + a2 + a3 - area) < 1e-9

def penalty(points):
    """
    Calculate penalty for points outside the triangle.
    """
    pen = 0.0
    for p in points:
        if not is_inside_triangle(p, V):
            # Distance to nearest edge? Or just a fixed penalty?
            # Fixed large penalty is simpler and effective for NM
            pen += 10.0
    return pen

def objective(flat_points):
    """
    Objective function for minimization.
    We want to MAXIMIZE min_area, so we MINIMIZE -min_area + penalty.
    """
    points = flat_points.reshape(N, 2)
    # Basic sanity check: if points are NaN or Inf
    if not np.all(np.isfinite(points)):
        return 1e6
    
    area = min_area(points)
    pen = penalty(points)
    return -area + pen

def generate_lattice_init():
    """
    Generate a structured initial guess based on a triangular lattice.
    """
    points = []
    # Try a 3x3 grid clipped to the triangle
    # Or specific symmetric points
    
    # Symmetric points:
    # Vertices
    points.append(V[0].copy())
    points.append(V[1].copy())
    points.append(V[2].copy())
    
    # Midpoints of edges
    points.append((V[0] + V[1]) / 2)
    points.append((V[1] + V[2]) / 2)
    points.append((V[2] + V[0]) / 2)
    
    # Centroid
    points.append(np.sum(V, axis=0) / 3)
    
    # We have 7 points. Need 4 more.
    # Place them in the middle of the "sub-triangles" formed by the medial triangle?
    # Or just add random points inside?
    # Let's add points at 1/3 and 2/3 along the medians?
    # Medians go from vertex to midpoint of opposite side.
    for i in range(3):
        v = V[i]
        # Midpoint of opposite side
        opp_mid = (V[(i+1)%3] + V[(i+2)%3]) / 2
        # Point at 1/3 from vertex
        p1 = v + (opp_mid - v) / 3
        points.append(p1)
        
    # We have 10 points. Need 1 more.
    # Maybe the center is already there. 
    # Let's just add a random point near the center for the 11th, or adjust the lattice.
    # Actually, the 1/3 points on medians might be close to each other.
    # Let's use a different strategy for the 4 extra points:
    # A smaller equilateral triangle rotated 180 degrees?
    
    # Let's restart the lattice generation with a more uniform distribution
    # using a grid in barycentric coordinates.
    points = []
    # Grid in barycentric coords (u, v, w) with u+v+w=1, u,v,w >= 0
    # For N=11, maybe a grid of 3 levels? 1, 3, 6? No.
    # Let's just use a random uniform start and a "grid" start.
    
    # Grid start:
    # Sample points on a fine grid and pick 11 that are well spread.
    # This is expensive to do "perfectly", so let's just use a simple heuristic.
    
    # For now, return the 7 symmetric + 4 random inside.
    rng = np.random.RandomState(42)
    for _ in range(4):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                p = u * V[0] + v * V[1] + (1 - u - v) * V[2]
                # Ensure it's not too close to existing points
                if all(np.linalg.norm(p - q) > 0.05 for q in points):
                    points.append(p)
                    break
    return np.array(points)

def generate_random_init(seed):
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                p = u * V[0] + v * V[1] + (1 - u - v) * V[2]
                points.append(p)
                break
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Initializations
    inits = [
        generate_random_init(1),
        generate_random_init(2),
        generate_lattice_init(),
    ]
    
    # Add a "perfect grid" like init if possible
    # For N=11, a 3x3 grid has 9 points. Add 2?
    # Let's try to create a 3x3 grid in barycentric coords.
    grid_points = []
    for i in range(3):
        for j in range(3):
            if i + j <= 2: # u=i/2, v=j/2, w=1-u-v
                # Normalize to 0..1 range for u,v,w summing to 1
                # Actually, let's just map i,j to barycentric
                u = i / 2.0
                v = j / 2.0
                if u + v <= 1.0:
                    p = u * V[0] + v * V[1] + (1 - u - v) * V[2]
                    grid_points.append(p)
    # We have 6 points from a coarse grid. 
    # Let's use a finer grid for initialization.
    
    # Run optimization from each init
    for idx, init in enumerate(inits):
        if time.time() - start_time > timeout:
            break
            
        try:
            res = minimize(
                objective, 
                init.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            
            opt_points = res.x.reshape(N, 2)
            
            # Verify validity and score
            if np.all(np.isfinite(opt_points)):
                # Check if all points are inside (with tolerance)
                inside = all(is_inside_triangle(p, V) for p in opt_points)
                if inside:
                    score = min_area(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points
                else:
                    # If not inside, we might still use it if penalty was low enough?
                    # No, task requires points on or inside.
                    # The penalty should have pushed them in, but let's clip.
                    # Clipping to triangle is tricky for arbitrary points.
                    # We'll just ignore invalid results.
                    pass
        except Exception:
            continue

    if best_points is None:
        # Fallback to a simple valid configuration
        best_points = generate_random_init(1)
        # Ensure they are inside
        # The random generator ensures this.
        
    return best_points

construct()
