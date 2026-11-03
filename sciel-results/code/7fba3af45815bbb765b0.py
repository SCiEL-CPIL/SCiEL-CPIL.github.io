import numpy as np
import time

N = 11
H = np.sqrt(3) / 2

def project_to_triangle(p):
    # Project point p onto the triangle (0,0), (1,0), (0.5, H)
    # Using barycentric coordinates or simple clipping?
    # Simple clipping is not enough for triangles.
    # We use the method of finding the closest point on the triangle.
    # Vertices: A=(0,0), B=(1,0), C=(0.5, H)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, H])
    
    def closest_point_on_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab

    # Check if p is inside
    # Barycentric coords
    v0 = C - A
    v1 = B - A
    v2 = p - A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    v = (d00 * d12 - d01 * d02) / denom
    w = (d11 * d02 - d01 * d12) / denom
    u = 1 - v - w
    
    if u >= -1e-9 and v >= -1e-9 and w >= -1e-9:
        return p
        
    # Outside, project to edges
    p_ab = closest_point_on_segment(p, A, B)
    p_bc = closest_point_on_segment(p, B, C)
    p_ca = closest_point_on_segment(p, C, A)
    
    d_ab = np.linalg.norm(p - p_ab)
    d_bc = np.linalg.norm(p - p_bc)
    d_ca = np.linalg.norm(p - p_ca)
    
    if d_ab < d_bc and d_ab < d_ca:
        return p_ab
    elif d_bc < d_ab and d_bc < d_ca:
        return p_bc
    else:
        return p_ca

def min_triangle_area_and_indices(points):
    min_a = float('inf')
    min_idx = (0, 1, 2)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_a:
                    min_a = area
                    min_idx = (i, j, k)
    return min_a, min_idx

def compute_gradient(points, idx):
    i, j, k = idx
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    # L = x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)
    # We want to maximize |L|/2.
    # Let's assume we are in a configuration where the orientation is fixed or we just follow the gradient of L.
    # If L > 0, grad L increases area. If L < 0, grad -L increases area.
    
    L = p_i[0] * (p_j[1] - p_k[1]) + p_j[0] * (p_k[1] - p_i[1]) + p_k[0] * (p_i[1] - p_j[1])
    
    grad = np.zeros((3, 2))
    
    # dL/dp_i
    grad_i = np.array([p_j[1] - p_k[1], p_k[0] - p_j[0]])
    # dL/dp_j
    grad_j = np.array([p_k[1] - p_i[1], p_i[0] - p_k[0]])
    # dL/dp_k
    grad_k = np.array([p_i[1] - p_j[1], p_j[0] - p_i[0]])
    
    if L < 0:
        grad_i = -grad_i
        grad_j = -grad_j
        grad_k = -grad_k
        
    grad[0] = grad_i
    grad[1] = grad_j
    grad[2] = grad_k
    
    return grad

def construct():
    start_time = time.time()
    time_limit = 115
    
    best_points_global = None
    best_score_global = -float('inf')
    
    num_runs = 20 # More runs, shorter each
    
    for run in range(num_runs):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization: Random points
        current_points = np.random.rand(N, 2)
        current_points[:, 0] *= 1.0
        current_points[:, 1] *= H
        # Project to triangle
        for i in range(N):
            current_points[i] = project_to_triangle(current_points[i])
            
        current_score, _ = min_triangle_area_and_indices(current_points)
        best_points_local = current_points.copy()
        best_score_local = current_score
        
        # Gradient Ascent Loop
        lr = 0.01
        no_improve_counter = 0
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            score, idx = min_triangle_area_and_indices(current_points)
            
            if score > best_score_local:
                best_score_local = score
                best_points_local = current_points.copy()
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                
            if no_improve_counter > 100:
                # Restart local search with perturbation
                current_points = best_points_local.copy()
                for i in range(N):
                    if np.random.rand() < 0.5:
                        current_points[i] += np.random.normal(0, 0.05, 2)
                        current_points[i] = project_to_triangle(current_points[i])
                no_improve_counter = 0
                continue
                
            grad = compute_gradient(current_points, idx)
            i, j, k = idx
            
            # Update points
            # Scale gradient by learning rate
            step_size = lr / (np.linalg.norm(grad) + 1e-6) * 0.01 # Normalize step
            
            # Apply update
            current_points[i] += grad[0] * step_size
            current_points[j] += grad[1] * step_size
            current_points[k] += grad[2] * step_size
            
            # Project back to triangle
            current_points[i] = project_to_triangle(current_points[i])
            current_points[j] = project_to_triangle(current_points[j])
            current_points[k] = project_to_triangle(current_points[k])
            
            # Decay learning rate slightly? Or keep constant.
            # lr *= 0.999 

        if best_score_local > best_score_global:
            best_score_global = best_score_local
            best_points_global = best_points_local.copy()
            
    return best_points_global
