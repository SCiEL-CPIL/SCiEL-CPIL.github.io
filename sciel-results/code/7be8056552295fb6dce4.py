import numpy as np
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    if not hasattr(min_triangle_area, "indices"):
        idxs = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    idxs.append((i, j, k))
        min_triangle_area.indices = np.array(idxs)
        
    idxs = min_triangle_area.indices
    p_i = points[idxs[:, 0]]
    p_j = points[idxs[:, 1]]
    p_k = points[idxs[:, 2]]
    
    dx_j = p_j[:, 0] - p_i[:, 0]
    dy_j = p_j[:, 1] - p_i[:, 1]
    dx_k = p_k[:, 0] - p_i[:, 0]
    dy_k = p_k[:, 1] - p_i[:, 1]
    
    areas = 0.5 * np.abs(dx_j * dy_k - dy_j * dx_k)
    return np.min(areas)

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    return (y >= -eps) and (y <= H + eps) and (x >= y/SQRT3 - eps) and (x <= 1 - y/SQRT3 + eps)

def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    # Clip y
    if y < 0:
        y = 0
    if y > H:
        y = H
    # Clip x based on y
    x_min = y / SQRT3
    x_max = 1 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H/2])
    points.append([0.75, H/2])
    
    # Center
    points.append([0.5, H/3])
    
    # Additional points to make 11
    # Add points along edges and interior
    # Bottom edge: add two more
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    
    # Interior points
    points.append([0.35, H/4])
    points.append([0.65, H/4])
    
    return np.array(points[:N])

def generate_random_points(seed):
    """Generate random points inside the triangle using rejection sampling."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(2)
        if u + v < 1:
            # Barycentric coordinates
            a = 1 - u - v
            b = u
            c = v
            x = b * 1.0 + c * 0.5
            y = c * H
            points.append([x, y])
    return np.array(points)

def local_optimize(points, max_iter=500, step_size=0.1, deadline=None):
    """Local optimization using coordinate descent / hill climbing."""
    points = points.copy()
    best_val = min_triangle_area(points)
    rng = np.random.RandomState(123)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            # Try moving point i in different directions
            best_local = points[i].copy()
            best_local_val = best_val
            
            for _ in range(20):  # Inner iterations for each point
                if deadline is not None and time.time() > deadline:
                    break
                    
                # Random perturbation
                dx = rng.uniform(-step_size, step_size, 2)
                new_p = best_local + dx
                new_p = project_to_triangle(new_p)
                
                # Check if it's still inside (with small tolerance)
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                    
                new_points = points.copy()
                new_points[i] = new_p
                
                val = min_triangle_area(new_points)
                if val > best_local_val + 1e-10:
                    best_local_val = val
                    best_local = new_p
                    improved = True
            
            points[i] = best_local
        
        # Update best value
        current_val = min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val
        
        # If no improvement in a full pass, reduce step size
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return points, best_val

def simulated_annealing(points, deadline, initial_temp=0.1, cooling=0.999, max_iter=5000):
    """Simulated annealing to escape local optima."""
    points = points.copy()
    best_points = points.copy()
    best_val = min_triangle_area(points)
    current_val = best_val
    temp = initial_temp
    rng = np.random.RandomState(456)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        # Pick a random point to move
        i = rng.randint(N)
        dx = rng.uniform(-0.05, 0.05, 2)
        new_p = points[i] + dx
        new_p = project_to_triangle(new_p)
        
        if not is_inside_triangle(new_p, eps=1e-6):
            continue
        
        new_points = points.copy()
        new_points[i] = new_p
        
        new_val = min_triangle_area(new_points)
        delta = new_val - current_val
        
        if delta > 0 or rng.random() < np.exp(delta / temp):
            points = new_points
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_points = points.copy()
        
        temp *= cooling
        if temp < 1e-8:
            break
    
    return best_points, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_points = None
    best_value = -1
    
    # Try multiple strategies
    strategies = [
        ("structured", generate_structured_initial()),
    ]
    
    # Add random initializations
    for seed in range(10):
        strategies.append((f"random_{seed}", generate_random_points(seed * 42 + 7)))
    
    for name, init_points in strategies:
        if time.time() > deadline:
            break
        
        # First, do local optimization
        opt_points, opt_val = local_optimize(init_points, max_iter=200, step_size=0.1, deadline=deadline)
        
        if opt_val > best_value:
            best_value = opt_val
            best_points = opt_points.copy()
        
        # If we have time, try simulated annealing from the best point
        if time.time() < deadline - 10:
            sa_points, sa_val = simulated_annealing(opt_points, deadline, initial_temp=0.05, cooling=0.9995, max_iter=5000)
            
            if sa_val > best_value:
                best_value = sa_val
                best_points = sa_points.copy()
    
    # Final local optimization pass on the best found
    if best_points is not None and time.time() < deadline - 5:
        final_points, final_val = local_optimize(best_points, max_iter=300, step_size=0.05, deadline=deadline)
        if final_val > best_value:
            best_value = final_val
            best_points = final_points.copy()
    
    # Verify all points are inside the triangle
    if best_points is not None:
        for i in range(len(best_points)):
            if not is_inside_triangle(best_points[i], eps=1e-6):
                best_points[i] = project_to_triangle(best_points[i])
    
    if best_points is None:
        # Fallback
        best_points = generate_structured_initial()
    
    return best_points
