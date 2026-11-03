"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])
CENTROID = (V1 + V2 + V3) / 3

def point_in_triangle(p, a=V1, b=V2, c=V3):
    """Check if point p is inside or on the triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def compute_all_areas(points):
    """Compute areas of all triangles formed by triples of points.
    Returns array of shape (n, n, n) where areas[i,j,k] is area of triangle (i,j,k)."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas[i,j,k] = area
                areas[j,i,k] = area
                areas[i,k,j] = area
                areas[k,j,i] = area
                areas[j,k,i] = area
                areas[k,i,j] = area
    return areas

def min_triangle_area(points):
    """Minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def initial_configuration(seed=42):
    """Create a structured initial configuration for 11 points in the triangle."""
    np.random.seed(seed)
    
    # Start with a hexagonal-like pattern: 3 rows of 4, 4, 3 or similar
    # Use barycentric coordinates for better distribution
    # Row 0 (bottom): 4 points near base
    # Row 1 (middle): 4 points
    # Row 2 (top): 3 points
    
    points = []
    
    # Bottom row: y ~ 0.1, spread across x
    y0 = 0.08
    for i in range(4):
        x = 0.15 + i * 0.233
        points.append(np.array([x, y0]))
    
    # Middle row: y ~ 0.35
    y1 = 0.35
    for i in range(4):
        x = 0.2 + i * 0.2
        points.append(np.array([x, y1]))
    
    # Top row: y ~ 0.55, 3 points
    y2 = 0.55
    for i in range(3):
        x = 0.3 + i * 0.2
        points.append(np.array([x, y2]))
    
    # Add small random perturbation
    for i in range(len(points)):
        points[i] += np.random.normal(0, 0.02, 2)
    
    # Ensure all points are inside triangle
    clipped = []
    for p in points:
        if not point_in_triangle(p):
            # Move toward centroid until inside
            t = 0.0
            while t < 1.0 and not point_in_triangle(CENTROID + t * (p - CENTROID)):
                t += 0.05
            p_new = CENTROID + t * (p - CENTROID)
            clipped.append(p_new)
        else:
            clipped.append(p)
    
    return np.array(clipped)

def simulated_annealing(points, max_iter=30000, T0=0.05, T_min=1e-7, cooling=0.999, step_init=0.04, seed=0):
    """Run simulated annealing to optimize the configuration."""
    np.random.seed(seed)
    current = points.copy()
    current_score = min_triangle_area(current)
    best = current.copy()
    best_score = current_score
    
    T = T0
    step_size = step_init
    no_improve = 0
    improve_threshold = 500
    
    for it in range(max_iter):
        if T < T_min:
            break
        
        # Adaptive step size: decrease if improving, increase if stuck
        if no_improve > improve_threshold:
            step_size = max(0.01, step_size * 0.95)
            no_improve = 0
        
        # Pick a random point to move
        idx = np.random.randint(0, N)
        old_point = current[idx].copy()
        
        # Random perturbation with adaptive step
        delta = np.random.normal(0, step_size, 2)
        new_point = old_point + delta
        
        # Check if new point is inside triangle
        if point_in_triangle(new_point):
            current[idx] = new_point
            new_score = min_triangle_area(current)
            
            # Accept if better or with probability exp((new-old)/T)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = current.copy()
                no_improve = 0
                # Slightly increase step size on improvement
                step_size = min(0.08, step_size * 1.01)
            else:
                prob = np.exp((new_score - current_score) / T)
                if np.random.random() < prob:
                    current_score = new_score
                else:
                    current[idx] = old_point
                no_improve += 1
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave some buffer
    
    # Multi-start with different seeds
    best_points = None
    best_score = 0.0
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time based on remaining time and number of seeds left
        seeds_left = len(seeds) - seeds.index(seed)
        time_per_start = max(5.0, remaining / seeds_left)
        
        # Determine max iterations based on available time
        # Each iteration takes roughly 0.1-0.2 ms for N=11
        max_iter = int(time_per_start * 5000)  # Rough estimate
        max_iter = max(1000, min(max_iter, 50000))
        
        points = initial_configuration(seed=seed)
        result, score = simulated_annealing(
            points, 
            max_iter=max_iter, 
            T0=0.05, 
            T_min=1e-7, 
            cooling=0.999,
            step_init=0.04,
            seed=seed
        )
        
        if score > best_score:
            best_score = score
            best_points = result
    
    return best_points if best_points is not None else initial_configuration(42)
# EVOLVE-BLOCK-END
