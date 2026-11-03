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

def project_to_triangle(p):
    """Project point p to the closest point inside the triangle."""
    if point_in_triangle(p):
        return p
    
    # If outside, find the closest point on the boundary.
    # This is a simplified projection. For robustness, we can use scipy if available, 
    # but let's stick to simple logic or just push inward.
    # A common simple heuristic: scale towards centroid until inside.
    centroid = (V1 + V2 + V3) / 3.0
    # Binary search for the closest point on the segment p-centroid that is inside.
    lo = 0.0
    hi = 1.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        p_mid = p + mid * (centroid - p)
        if point_in_triangle(p_mid):
            hi = mid
        else:
            lo = mid
    return p + hi * (centroid - p)

def min_triangle_info(points):
    """Returns the minimum area and the indices of the triangle forming it."""
    n = len(points)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    for i, j, k in combinations(range(n), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        if area < min_area:
            min_area = area
            min_idx = (i, j, k)
    return min_area, min_idx

def compute_gradient(points, idx):
    """Compute the gradient of the area of the triangle formed by points at indices idx."""
    i, j, k = idx
    pi, pj, pk = points[i], points[j], points[k]
    
    # Vector from i to j, i to k
    v1 = pj - pi
    v2 = pk - pi
    
    # Area is 0.5 * |v1 x v2|
    # We assume the orientation is such that we want to maximize the signed area 
    # or we just maximize the absolute area. 
    # The gradient of the absolute area points in the direction that increases the area.
    # If the triangle is degenerate, gradient is undefined, but we handle that.
    
    # Let's compute gradients for pi, pj, pk assuming we maximize the signed area 
    # S = 0.5 * (v1x*v2y - v1y*v2x).
    # If the signed area is negative, we flip the sign of the gradient to maximize abs area.
    
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    
    # Gradients for S (signed)
    # dS/dpi = -0.5 * ( (y_j - y_i + y_k - y_i), (x_j - x_i + x_k - x_i) ) ... wait, derived before.
    # dS/dpi_x = -0.5 * (v2y + v1y) = -0.5 * (y_k - y_i + y_j - y_i)
    # dS/dpi_y = 0.5 * (v2x + v1x) = 0.5 * (x_k - x_i + x_j - x_i)
    
    # dS/dpj_x = 0.5 * v2y = 0.5 * (y_k - y_i)
    # dS/dpj_y = -0.5 * v2x = -0.5 * (x_k - x_i)
    
    # dS/dpk_x = -0.5 * v1y = -0.5 * (y_j - y_i)
    # dS/dpk_y = 0.5 * v1x = 0.5 * (x_j - x_i)
    
    grad = np.zeros_like(points)
    
    if abs(cross) < 1e-12:
        return grad, 0.0
        
    # We want to maximize the area. The direction of the gradient of the signed area
    # depends on the orientation.
    # If cross > 0, increasing S increases Area.
    # If cross < 0, decreasing S increases Area (since Area = -S).
    # So, if cross < 0, we flip the gradient direction.
    
    sign_factor = 1.0 if cross > 0 else -1.0
    
    grad[i][0] = sign_factor * (-0.5 * (v2[1] + v1[1]))
    grad[i][1] = sign_factor * (0.5 * (v2[0] + v1[0]))
    
    grad[j][0] = sign_factor * (0.5 * v2[1])
    grad[j][1] = sign_factor * (-0.5 * v2[0])
    
    grad[k][0] = sign_factor * (-0.5 * v1[1])
    grad[k][1] = sign_factor * (0.5 * v1[0])
    
    return grad, min_area

def gradient_ascent(points, max_iter=200, tol=1e-6):
    """Perform gradient ascent to maximize the minimum triangle area."""
    current_points = points.copy()
    current_min_area, _ = min_triangle_info(current_points)
    
    step_size = 0.01
    for it in range(max_iter):
        min_area, min_idx = min_triangle_info(current_points)
        
        # If no improvement possible or converged
        if min_area - current_min_area < tol:
             # Check if we can still improve with a smaller step? 
             # For simplicity, break if step is too small or no gain.
             if step_size < 1e-4:
                 break
             step_size *= 0.5
             continue

        grad, _ = compute_gradient(current_points, min_idx)
        
        # Normalize gradient to prevent large jumps
        norm = np.linalg.norm(grad)
        if norm < 1e-12:
            break
        grad = grad / norm
        
        # Try to update
        new_points = current_points + step_size * grad
        
        # Project points back to triangle
        for i in range(len(new_points)):
            if not point_in_triangle(new_points[i]):
                new_points[i] = project_to_triangle(new_points[i])
        
        new_min_area, _ = min_triangle_info(new_points)
        
        if new_min_area > min_area:
            current_points = new_points
            current_min_area = new_min_area
            step_size *= 1.05 # Slightly increase step size if we made progress
            if step_size > 0.1:
                step_size = 0.1
        else:
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    return current_points

def construct():
    np.random.seed(42)
    start_time = time.time()
    best_points = None
    best_score = 0.0
    
    # Run multiple starts
    # Start 1: Centered grid
    # Start 2: Random
    # Start 3: Vertices + Centroid + Random
    
    starts = []
    
    # Start 1: Simple grid
    p1 = np.array([
        [0.2, 0.1], [0.4, 0.1], [0.6, 0.1], [0.8, 0.1],
        [0.3, 0.3], [0.5, 0.3], [0.7, 0.3],
        [0.4, 0.5], [0.6, 0.5],
        [0.5, 0.7],
        [0.5, 0.0] # Corner? No, let's use centroid
    ])
    # Ensure inside
    p1 = np.array([project_to_triangle(p) for p in p1])
    starts.append(p1)
    
    # Start 2: Random
    p2 = np.random.rand(11, 2)
    # Scale to triangle roughly
    # Uniform sampling in triangle:
    # u, v in [0,1]. If u+v > 1, reflect.
    # P = V1 + u*(V2-V1) + v*(V3-V1)
    for _ in range(10):
        u = np.random.rand(11)
        v = np.random.rand(11)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        p2 = V1 + u[:, None]*(V2-V1) + v[:, None]*(V3-V1)
        starts.append(p2)
        
    # Start 3: Vertices and Centroid
    p3 = np.array([
        V1, V2, V3,
        (V1+V2)/2, (V2+V3)/2, (V3+V1)/2,
        (V1+V2+V3)/3,
        V1 + 0.25*(V2-V1) + 0.25*(V3-V1),
        V1 + 0.5*(V2-V1) + 0.1*(V3-V1),
        V1 + 0.1*(V2-V1) + 0.5*(V3-V1),
        V1 + 0.4*(V2-V1) + 0.4*(V3-V1)
    ])
    starts.append(p3)
    
    for start in starts:
        if time.time() - start_time > 100: # Leave time for final check
            break
        optimized = gradient_ascent(start)
        score, _ = min_triangle_info(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized
            
    if best_points is None:
        return starts[0]
        
    return best_points
# EVOLVE-BLOCK-END
