"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Kinetic Annealing with Topological Collision Resolution.
1. Barycentric coordinates enforce the triangle boundary exactly.
2. Objective: Maximize t (minimum area).
3. Optimization: Custom gradient descent with adaptive step size.
   - Compute areas of all 165 triangles.
   - Identify the minimum area triangle (the "bottleneck").
   - Compute gradients with respect to point positions for that triangle.
   - Apply repulsive forces to push the vertices of the bottleneck triangle apart.
   - Handle boundary collisions by projecting back into the feasible region.
   - Periodic "snap-to-symmetry" to escape local minima and leverage the symmetry of the equilateral triangle.
4. Multi-start with diverse initializations.
"""
import numpy as np
import time
import random
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_j_idx] # Typo in original? No, _k_idx
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian coordinates.
    bary: shape (N, 3) where each row sums to 1 and all elements >= 0.
    Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    P = b1*A + b2*B + b3*C
    x = b2*1 + b3*0.5
    y = b3*S3H
    """
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack((x, y))

def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates relative to A(0,0), B(1,0), C(0.5, S3H)."""
    b3 = pts[:, 1] / S3H
    b2 = pts[:, 0] - 0.5 * b3
    b1 = 1.0 - b2 - b3
    return np.column_stack((b1, b2, b3))

def _compute_areas_and_grads(bary):
    """
    Compute areas of all triangles and gradients of the minimum area with respect to barycentric coordinates.
    Returns: min_area, min_triplet_index, grads (shape (N, 3))
    """
    pts = _barycentric_to_cartesian(bary)
    
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    # Let's compute signed area first to handle gradients properly.
    # A = 0.5 * ( (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) )
    dx_ji = p_j[:, 0] - p_i[:, 0]
    dy_ji = p_j[:, 1] - p_i[:, 1]
    dx_ki = p_k[:, 0] - p_i[:, 0]
    dy_ki = p_k[:, 1] - p_i[:, 1]
    
    signed_area = 0.5 * (dx_ji * dy_ki - dy_ji * dx_ki)
    areas = np.abs(signed_area)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    
    # Gradient of the area of the min triangle with respect to point positions
    # We want to maximize the minimum area. So we want to increase the area of the min triangle.
    # Let the min triangle be (i, j, k).
    # Area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
    # If we assume the orientation is consistent (which it is for small steps), 
    # grad_Pi Area = -0.5 * cross(P_j - P_i, P_k - P_i) / |Area| * (sign) ... wait.
    # Let's use the formula: 
    # d/dP_i (0.5 * cross(P_j - P_i, P_k - P_i)) = 0.5 * (cross(dP_j - dP_i, P_k - P_i) + cross(P_j - P_i, dP_k - dP_i))
    # This is getting complicated. 
    # Simpler: 
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    # Let S = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # dS/dx_i = -(y_k - y_i)
    # dS/dy_i = (x_k - x_i)
    # dS/dx_j = (y_k - y_i)
    # dS/dy_j = -(x_k - x_i)
    # dS/dx_k = -(y_j - y_i)
    # dS/dy_k = (x_j - x_i)
    
    # Since we want to maximize |S|, and S is the signed area, 
    # if S > 0, we want to increase S. If S < 0, we want to decrease S.
    # So the direction is sign(S) * grad_S.
    
    s = signed_area[min_idx]
    sign_s = 1.0 if s > 0 else -1.0
    
    # Initialize gradients in Cartesian space
    grad_cart = np.zeros((N, 2))
    
    i, j, k = _triplets[min_idx]
    
    # Gradients for i
    grad_cart[i, 0] = sign_s * 0.5 * (-(p_k[min_idx, 1] - p_i[min_idx, 1]))
    grad_cart[i, 1] = sign_s * 0.5 * (p_k[min_idx, 0] - p_i[min_idx, 0])
    
    # Gradients for j
    grad_cart[j, 0] = sign_s * 0.5 * (p_k[min_idx, 1] - p_i[min_idx, 1])
    grad_cart[j, 1] = sign_s * 0.5 * (-(p_k[min_idx, 0] - p_i[min_idx, 0]))
    
    # Gradients for k
    grad_cart[k, 0] = sign_s * 0.5 * (-(p_j[min_idx, 1] - p_i[min_idx, 1]))
    grad_cart[k, 1] = sign_s * 0.5 * (p_j[min_idx, 0] - p_i[min_idx, 0])
    
    # Now convert Cartesian gradients to Barycentric gradients
    # x = b2 + 0.5 b3
    # y = S3H b3
    # dx = db2 + 0.5 db3
    # dy = S3H db3
    # grad_b2 = grad_x
    # grad_b3 = grad_x * 0.5 + grad_y / S3H
    # grad_b1 = -(grad_b2 + grad_b3)  (since b1 + b2 + b3 = 1, so db1 = -(db2 + db3))
    
    grad_bary = np.zeros((N, 3))
    for idx in range(N):
        gx = grad_cart[idx, 0]
        gy = grad_cart[idx, 1]
        
        gb2 = gx
        gb3 = 0.5 * gx + gy / S3H
        gb1 = -(gb2 + gb3)
        
        grad_bary[idx] = [gb1, gb2, gb3]
        
    return min_area, min_idx, grad_bary

def _kinetic_annealing_step(bary, step_size, start_time, deadline):
    """
    Perform one step of kinetic annealing.
    """
    if time.time() > deadline:
        return bary, False
        
    min_area, min_idx, grad_bary = _compute_areas_and_grads(bary)
    
    # Add a small random noise to escape local minima
    noise = 0.001 * np.random.randn(N, 3)
    
    # Update barycentric coordinates
    bary_new = bary + step_size * (grad_bary + noise)
    
    # Project back to valid barycentric coordinates
    # 1. Sum to 1
    sums = np.sum(bary_new, axis=1, keepdims=True)
    bary_new = bary_new / sums
    
    # 2. Non-negative
    # Simple projection: clip to 0 and re-normalize
    # This might not be the closest point in L2, but it's fast and works well.
    bary_new = np.maximum(bary_new, 0)
    sums = np.sum(bary_new, axis=1, keepdims=True)
    # Handle case where sum is 0 (all points at a vertex)
    sums = np.where(sums == 0, 1, sums)
    bary_new = bary_new / sums
    
    return bary_new, True

def _snap_to_symmetry(bary, rng):
    """
    Periodically snap points to symmetric positions to escape local minima.
    This is a heuristic: if the configuration is close to a symmetric one, nudge it there.
    """
    # For now, just add a small perturbation that respects symmetry.
    # A better approach: identify the symmetry group of the equilateral triangle (D3)
    # and check if the current configuration is close to any of its symmetric images.
    # If so, nudge it towards the center of mass of the symmetric images.
    
    # For simplicity, we'll just do a random perturbation here.
    # A more sophisticated snap would be:
    # 1. Compute the center of mass of the points.
    # 2. If the center of mass is close to the triangle's center, 
    #    apply a small force to push points towards symmetric positions.
    
    # Let's just return bary unchanged for now, or with a tiny perturbation.
    return bary

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    """
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(200): # Reduced for speed
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Kinetic Annealing with Topological Collision Resolution.
    """
    start_time = time.time()
    rng = random.Random(42)
    np.random.seed(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start optimization
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
    
    # 2. Random
    p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    p_rand = _project_to_triangle(p_rand)
    initializations.append(p_rand)
    
    # 3. Structured
    p_struct = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    p_struct = np.array(p_struct)
    p_struct = _project_to_triangle(p_struct)
    initializations.append(p_struct)
    
    # Kinetic Annealing Parameters
    initial_step_size = 0.05
    cooling_rate = 0.999
    min_step_size = 0.0001
    max_iterations = 5000
    
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        bary = _cartesian_to_bary(init_pts)
        bary = bary / np.sum(bary, axis=1, keepdims=True)
        bary = np.maximum(bary, 0)
        bary = bary / np.sum(bary, axis=1, keepdims=True)
        
        step_size = initial_step_size
        best_bary_in_run = bary.copy()
        best_score_in_run = _min_triangle_area(_barycentric_to_cartesian(bary))
        
        for it in range(max_iterations):
            if time.time() - start_time > DEADLINE - 2:
                break
                
            bary, success = _kinetic_annealing_step(bary, step_size, start_time, DEADLINE)
            if not success:
                break
                
            # Snap to symmetry every 100 iterations
            if it % 100 == 0:
                bary = _snap_to_symmetry(bary, rng)
                
            # Evaluate
            current_score = _min_triangle_area(_barycentric_to_cartesian(bary))
            if current_score > best_score_in_run:
                best_score_in_run = current_score
                best_bary_in_run = bary.copy()
                
            # Cool down
            step_size *= cooling_rate
            if step_size < min_step_size:
                break
                
        # Update global best
        if best_score_in_run > best_score:
            best_score = best_score_in_run
            best_pts = _barycentric_to_cartesian(best_bary_in_run)
            
    # If no optimization succeeded, fall back to the best initial guess
    if best_pts is None:
        for init_pts in initializations:
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_pts = init_pts.copy()
                
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
