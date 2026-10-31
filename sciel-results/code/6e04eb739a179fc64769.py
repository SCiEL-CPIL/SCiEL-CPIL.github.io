"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from itertools import combinations

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])
AREA = np.sqrt(3)/4  # area of the triangle

def bary_to_cart(b):
    """Convert barycentric coordinates (a, b, c) with a+b+c=1 to Cartesian."""
    a, b, c = b
    return a * V0 + b * V1 + c * V2

def cart_to_bary(p):
    """Convert Cartesian point to barycentric coordinates."""
    # Solve p = a*V0 + b*V1 + c*V2, a+b+c=1
    # a = 1 - b - c
    # p = (1-b-c)*V0 + b*V1 + c*V2
    # p - V0 = b*(V1-V0) + c*(V2-V0)
    # [p_x - V0_x, p_y - V0_y] = b*[V1_x-V0_x, V1_y-V0_y] + c*[V2_x-V0_x, V2_y-V0_y]
    dx = p[0] - V0[0]
    dy = p[1] - V0[1]
    # V1-V0 = [1, 0], V2-V0 = [0.5, sqrt(3)/2]
    # dx = b + 0.5*c
    # dy = (sqrt(3)/2)*c
    c = dy / (np.sqrt(3)/2)
    b = dx - 0.5 * c
    a = 1 - b - c
    return np.array([a, b, c])

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the triangle."""
    b = cart_to_bary(p)
    return all(bi >= -eps for bi in b)

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, divided by triangle area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area / AREA

def random_bary_init(n, rng):
    """Generate n random points in barycentric coordinates inside the triangle."""
    points_bary = []
    for _ in range(n):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        points_bary.append(np.array([1-u-v, u, v]))
    return points_bary

def objective(points_cart):
    """The objective: minimum triangle area / triangle area."""
    return min_triangle_area(points_cart)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Start with a good initial configuration
    # Use a grid-like initial placement for better starting point
    # Place points in a structured way first
    points_bary = []
    # Try a few different initial patterns
    best_points = None
    best_score = 0
    
    # Pattern 1: Random
    for trial in range(5):
        pb = random_bary_init(N, rng)
        pc = np.array([bary_to_cart(b) for b in pb])
        score = objective(pc)
        if score > best_score:
            best_score = score
            best_points = pc.copy()
    
    # Pattern 2: Grid-like in barycentric
    # Create a grid in the triangle
    for trial in range(3):
        pb = []
        # Use a simple stratified sampling
        for i in range(N):
            u = (i + 0.5) / N
            v = rng.random() * (1 - u)
            pb.append(np.array([1-u-v, u, v]))
        pc = np.array([bary_to_cart(b) for b in pb])
        score = objective(pc)
        if score > best_score:
            best_score = score
            best_points = pc.copy()
    
    if best_points is None:
        best_points = np.array([bary_to_cart(b) for b in random_bary_init(N, rng)])
        best_score = objective(best_points)
    
    # Simulated annealing
    current_points = best_points.copy()
    current_score = best_score
    T = 0.01
    T_min = 1e-6
    alpha = 0.999
    step_size = 0.05
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = rng.integers(0, N)
        old_pos = current_points[idx].copy()
        
        # Move in a random direction
        direction = rng.normal(size=2)
        direction /= (np.linalg.norm(direction) + 1e-12)
        new_pos = old_pos + step_size * direction
        
        # Check if new position is inside triangle
        if not is_inside_triangle(new_pos):
            continue
        
        current_points[idx] = new_pos
        new_score = objective(current_points)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            current_points[idx] = old_pos
        
        T *= alpha
        if T < T_min:
            T = T_min
        
        # Occasionally restart with a new random configuration
        if rng.random() < 0.001:
            pb = random_bary_init(N, rng)
            current_points = np.array([bary_to_cart(b) for b in pb])
            current_score = objective(current_points)
            T = 0.01
    
    # Final safety clamping: ensure all points are strictly inside
    for i in range(len(best_points)):
        p = best_points[i]
        b = cart_to_bary(p)
        # Clamp barycentric coordinates to be >= 0
        if np.any(b < 0):
            b = np.clip(b, 0, None)
            b /= b.sum()
            best_points[i] = bary_to_cart(b)
    
    # Verify all points are inside
    for i in range(len(best_points)):
        if not is_inside_triangle(best_points[i]):
            # Fallback: place at centroid
            best_points[i] = (V0 + V1 + V2) / 3
    
    return best_points
