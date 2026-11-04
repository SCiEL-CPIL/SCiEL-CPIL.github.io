import numpy as np
import time
from itertools import combinations

N = 11
S3 = np.sqrt(3.0)

def _inside_triangle(p):
    x, y = p
    if y < -1e-9:
        return False
    if y > S3 * x + 1e-9:
        return False
    if y > -S3 * (x - 1.0) + 1e-9:
        return False
    return True

def _project_to_triangle(p):
    """Project point p to the nearest point inside the triangle using simple barycentric clipping."""
    x, y = p
    
    # Barycentric coordinates
    # A=(0,0), B=(1,0), C=(0.5, S3/2)
    # v0 = B - A = (1, 0)
    # v1 = C - A = (0.5, S3/2)
    # v2 = P - A = (x, y)
    
    dot00 = 1.0
    dot01 = 0.5
    dot02 = x
    dot11 = 0.25 + 3/4 = 1.0
    dot12 = 0.5*x + S3/2*y
    
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # Clip to [0, 1] and u+v <= 1
    if u < 0:
        u = 0
    if v < 0:
        v = 0
    if u + v > 1:
        s = 1 / (u + v)
        u *= s
        v *= s
        
    # Convert back to cartesian
    px = u * 1.0 + v * 0.5
    py = v * S3/2
    return np.array([px, py])

def _min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial configuration: Grid-like
    # 11 points in a 3x3 grid + 2 extra?
    # Or just random with boundary allowed.
    
    # Let's use a set of known good starting points:
    # Vertices, edge midpoints, and center.
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3/2],
        [0.5, 0.0],
        [0.25, S3/4],
        [0.75, S3/4],
        [0.5, S3/6],
        [0.125, S3/12],
        [0.875, S3/12],
        [0.375, S3/4], # Adjusted
        [0.625, S3/4]  # Adjusted
    ]
    
    # Ensure all are inside
    points = np.array([_project_to_triangle(p) for p in pts])
    
    current_area = _min_triangle_area(points)
    best_points = points.copy()
    best_area = current_area
    
    step_size = 0.05
    no_improve = 0
    max_no_improve = 20
    
    while time.time() < deadline:
        improved = False
        
        # Find worst triangle
        min_area = float('inf')
        worst_idx = None
        for i, j, k in combinations(range(N), 3):
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            area = 0.5 * abs(cross)
            if area < min_area:
                min_area = area
                worst_idx = (i, j, k)
        
        # Try to move points to increase min_area
        # Strategy: Move each point of the worst triangle away from the others
        # Or move any point in a random direction if it increases the global min area
        
        for _ in range(50):
            if time.time() > deadline:
                break
            
            trial = points.copy()
            idx = np.random.randint(N)
            
            # Random perturbation
            angle = np.random.uniform(0, 2*np.pi)
            trial[idx][0] += step_size * np.cos(angle)
            trial[idx][1] += step_size * np.sin(angle)
            
            # Project to triangle
            trial[idx] = _project_to_triangle(trial[idx])
            
            new_area = _min_triangle_area(trial)
            
            if new_area > current_area + 1e-12:
                current_area = new_area
                points = trial
                improved = True
                no_improve = 0
                if current_area > best_area:
                    best_area = current_area
                    best_points = points.copy()
            else:
                no_improve += 1
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-8:
                break
        else:
            # If improved, keep step size or increase slightly?
            # Keep it same for stability.
            pass
        
        if no_improve > max_no_improve:
            # Restart with larger step?
            step_size = 0.1
            no_improve = 0
            # Maybe randomize a bit?
            for i in range(N):
                angle = np.random.uniform(0, 2*np.pi)
                points[i][0] += 0.05 * np.cos(angle)
                points[i][1] += 0.05 * np.sin(angle)
                points[i] = _project_to_triangle(points[i])
            current_area = _min_triangle_area(points)
        
        if time.time() > deadline - 5:
            break
    
    return best_points
