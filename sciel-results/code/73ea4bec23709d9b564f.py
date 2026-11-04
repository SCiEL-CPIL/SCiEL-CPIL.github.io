import numpy as np
import time

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

# Pre-compute all triangle indices
tri_indices = []
for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            tri_indices.append((i, j, k))
tri_indices = np.array(tri_indices)

def min_area_vectorized(pts):
    """
    Compute minimum triangle area for N points, vectorized.
    pts: (N, 2) array
    """
    pi = pts[tri_indices[:, 0]]
    pj = pts[tri_indices[:, 1]]
    pk = pts[tri_indices[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def project_points_to_triangle(pts):
    """
    Project points to the triangle if outside.
    """
    v0 = B - A
    v1 = C - A
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    
    denom = d00 * d11 - d01 * d01
    
    v2 = pts - A
    
    d02 = np.sum(v0 * v2, axis=1)
    d12 = np.sum(v1 * v2, axis=1)
    
    v = (d11 * d02 - d01 * d12) / denom
    w = (d00 * d12 - d01 * d02) / denom
    u = 1 - v - w
    
    # If any barycentric coord is negative, the point is outside.
    # We clamp them to 0 and renormalize to keep points inside.
    # This is a simple projection to the convex hull (triangle).
    inside = (u >= -1e-10) & (v >= -1e-10) & (w >= -1e-10)
    
    projected = pts.copy()
    for i in range(N):
        if not inside[i]:
            u_i = max(0.0, u[i])
            v_i = max(0.0, v[i])
            w_i = max(0.0, w[i])
            sum_uvw = u_i + v_i + w_i
            if sum_uvw == 0:
                # Fallback to centroid if everything is 0 (shouldn't happen)
                projected[i] = (A + B + C) / 3.0
            else:
                u_i /= sum_uvw
                v_i /= sum_uvw
                w_i /= sum_uvw
                projected[i] = u_i * A + v_i * B + w_i * C
    return projected

def random_point_in_triangle(rng):
    """Generate a random point inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    return u * A + v * B + (1 - u - v) * C

def construct():
    """
    Main function: find 11 points in the unit triangle maximizing minimum triangle area.
    Uses Simulated Annealing.
    """
    rng = np.random.default_rng(42)
    
    # Initial configuration: random points
    current_pts = np.array([random_point_in_triangle(rng) for _ in range(N)])
    current_score = min_area_vectorized(current_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    start_temp = 0.01
    end_temp = 1e-6
    num_iterations = 50000
    time_limit = 110.0 # seconds
    start_time = time.time()
    
    # Precompute number of iterations based on time if possible, but fixed is safer
    # We'll run a fixed number of iterations but break if time limit is hit.
    
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Temperature schedule
        # Linear cooling or geometric? Geometric is common.
        # T = T_start * (T_end / T_start)^(i / num_iterations)
        ratio = (end_temp / start_temp) ** (i / num_iterations)
        temp = start_temp * ratio
        
        # Perturbation
        # Move a random point by a small random amount
        # Or move all points slightly? Moving one point is faster.
        idx = rng.integers(0, N)
        
        # Step size decreases with temperature
        step_size = 0.05 * temp / start_temp + 0.001
        
        # Random direction
        angle = rng.random() * 2 * np.pi
        displacement = np.array([np.cos(angle), np.sin(angle)]) * step_size
        
        new_pts = current_pts.copy()
        new_pts[idx] += displacement
        
        # Project to triangle
        new_pts = project_points_to_triangle(new_pts)
        
        new_score = min_area_vectorized(new_pts)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / temp):
            current_pts = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                
    return best_pts
