import numpy as np
import time
import itertools

N = 11
# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triangles
_indices = np.array(list(itertools.combinations(range(N), 3)))

def _get_barycentric(points):
    """
    Convert Cartesian points to barycentric coordinates (u, v) where w=1-u-v.
    p = A + u*(B-A) + v*(C-A)
    """
    # B-A = [1, 0]
    # C-A = [0.5, sqrt(3)/2]
    # p - A = [x, y]
    # x = u + 0.5v
    # y = (sqrt(3)/2)v  => v = 2y/sqrt(3)
    # u = x - 0.5v = x - y/sqrt(3)
    x = points[:, 0]
    y = points[:, 1]
    v = 2.0 * y / np.sqrt(3.0)
    u = x - 0.5 * v
    return np.column_stack((u, v))

def _get_cartesian(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian points.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = (np.sqrt(3.0) / 2.0) * v
    return np.column_stack((x, y))

def _is_valid_bary(bary):
    """Check if barycentric coordinates are inside the triangle."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    return np.all(u >= 0) and np.all(v >= 0) and np.all(w >= 0)

def _min_area(bary):
    """
    Calculate the minimum area of triangles formed by points defined by barycentric coords.
    Area = 0.5 * |det([p2-p1, p3-p1])|
    In barycentric coords, the area is proportional to the determinant of the matrix of coordinates.
    Specifically, if points are p_i, p_j, p_k with barycentric coords (u_i, v_i), etc.
    The area is (Area of ABC) * |det([[u_i, v_i, 1], [u_j, v_j, 1], [u_k, v_k, 1]])| / 2 ?
    Actually, the area of a triangle with vertices (u1,v1), (u2,v2), (u3,v3) in barycentric space
    is 0.5 * |det([ [u1, v1, 1], [u2, v2, 1], [u3, v3, 1] ])| * Area(ABC).
    Let's verify.
    p1 = u1*B + v1*C (since A=0) -> No, p = u(B-A) + v(C-A) + A.
    Vector p2-p1 = (u2-u1)(B-A) + (v2-v1)(C-A)
    Vector p3-p1 = (u3-u1)(B-A) + (v3-v1)(C-A)
    Cross product (in 2D, z-component) = det([p2-p1, p3-p1])
    = det([(u2-u1)(B-A) + (v2-v1)(C-A), (u3-u1)(B-A) + (v3-v1)(C-A)])
    = (u2-u1)(v3-v1) det(B-A, C-A) + (v2-v1)(u3-u1) det(C-A, B-A)
    det(B-A, C-A) = 1 * (sqrt(3)/2) - 0 * 0.5 = sqrt(3)/2
    det(C-A, B-A) = -sqrt(3)/2
    So Cross = (sqrt(3)/2) * [ (u2-u1)(v3-v1) - (v2-v1)(u3-u1) ]
    Area = 0.5 * |Cross| = (sqrt(3)/4) * | (u2-u1)(v3-v1) - (v2-v1)(u3-u1) |
    The term in brackets is the determinant of [[u2-u1, v2-v1], [u3-u1, v3-v1]]
    which is the same as det([[1, u2, v2], [1, u3, v3], [0, 0, 0]]... no.
    It is det([[u2, v2, 1], [u3, v3, 1], [u1, v1, 1]])?
    Let M = [[u1, v1, 1], [u2, v2, 1], [u3, v3, 1]]
    det(M) = u1(v2-v3) - v1(u2-u3) + (u2v3 - u3v2)
    = u1v2 - u1v3 - v1u2 + v1u3 + u2v3 - u3v2
    Expand (u2-u1)(v3-v1) - (v2-v1)(u3-u1):
    = u2v3 - u2v1 - u1v3 + u1v1 - (v2u3 - v2u1 - v1u3 + v1u1)
    = u2v3 - u2v1 - u1v3 + u1v1 - v2u3 + v2u1 + v1u3 - v1u1
    = u2v3 - u1v2 - u1v3 + v1u3 + u1v2 - v2u3  (cancels u1v1, -v1u1, -u2v1+v2u1)
    Wait, u2v1 = v1u2.
    Term: -u2v1 + v2u1 = -v1u2 + u1v2.
    So sum = u2v3 - u1v3 + v1u3 - v2u3 + (-v1u2 + u1v2)
    Compare with det(M):
    det(M) = u1v2 - u1v3 - v1u2 + v1u3 + u2v3 - u3v2
    = u1v2 - u1v3 - v1u2 + v1u3 + u2v3 - v2u3
    They are identical.
    So Area = (sqrt(3)/4) * |det(M)|.
    Since we want to maximize the minimum area, we can just maximize the minimum |det(M)|.
    The constant factor doesn't change the optimization.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Extract coordinates for each triangle
    i, j, k = _indices.T
    ui, vi = u[i], v[i]
    uj, vj = u[j], v[j]
    uk, vk = u[k], v[k]
    
    # Calculate determinant
    # det = ui*(vj-vk) - vi*(uj-uk) + (uj*vk - uk*vj)
    dets = ui * (vj - vk) - vi * (uj - uk) + (uj * vk - uk * vj)
    
    min_det = np.min(np.abs(dets))
    return min_det

def _objective(bary):
    """Objective function: negative of min area determinant."""
    if not _is_valid_bary(bary):
        # Penalty for being outside
        u = bary[:, 0]
        v = bary[:, 1]
        w = 1.0 - u - v
        # Distance to boundary
        # Simple penalty
        penalty = np.sum(np.maximum(0, -u)**2 + np.maximum(0, -v)**2 + np.maximum(0, -w)**2)
        return 1000.0 + penalty
    
    return -_min_area(bary)

def _clip_bary(bary):
    """Clip barycentric coordinates to the triangle."""
    u = bary[:, 0]
    v = bary[:, 1]
    
    # If u < 0, set u=0 and adjust v? No, projection is complex.
    # For simulated annealing, we can just reject invalid moves or project.
    # Projection onto triangle:
    # 1. If u < 0, set u=0. Then check if v is valid.
    #    If v < 0, set v=0.
    #    If v > 1, set v=1? No, if u=0, v can be up to 1.
    # 2. If v < 0, set v=0.
    # 3. If u+v > 1, scale down? Or project to edge u+v=1.
    
    # Simple clipping strategy:
    new_u = np.clip(u, 0, 1)
    new_v = np.clip(v, 0, 1)
    # Check u+v <= 1
    s = new_u + new_v
    mask = s > 1
    if np.any(mask):
        # Scale u and v so that u+v=1
        scale = 1.0 / s[mask]
        new_u[mask] *= scale
        new_v[mask] *= scale
        
    return np.column_stack((new_u, new_v))

def _generate_initial_bary(rng):
    """Generate a random valid barycentric configuration."""
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[i] = [u, v]
                break
    return bary

def _generate_structured_bary():
    """Generate a structured initial guess (e.g. grid-like)."""
    # Try a 3x3 grid + 2 points?
    # Or just use a known good configuration if available, but we don't have one.
    # Let's try to place points on a triangular lattice.
    # For 11 points, maybe 1 center, 6 around, 4 corners?
    # Corners: (0,0), (1,0), (0,1) in barycentric.
    # Midpoints: (0.5, 0), (0, 0.5), (0.5, 0.5)
    # Center: (1/3, 1/3)
    # That's 7 points.
    # Add 4 more? Maybe midpoints of the segments from center to corners?
    # Or just random is fine.
    
    # Let's return a random one to keep it simple, or a mix.
    rng = np.random.default_rng(123)
    return _generate_initial_bary(rng)

def _simulated_annealing(bary0, time_budget, rng):
    """Run simulated annealing starting from bary0."""
    current_bary = bary0.copy()
    current_obj = _objective(current_bary)
    
    best_bary = current_bary.copy()
    best_obj = current_obj
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    T = 0.1 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate neighbor
        # Randomly choose a move type
        move_type = rng.integers(0, 3)
        
        new_bary = current_bary.copy()
        
        if move_type == 0:
            # Local jitter: perturb one point
            idx = rng.integers(0, N)
            # Gaussian noise
            noise = rng.normal(0, 0.05, 2)
            new_bary[idx] += noise
        elif move_type == 1:
            # Swap: swap coordinates of two points
            i, j = rng.choice(N, size=2, replace=False)
            new_bary[i], new_bary[j] = new_bary[j].copy(), new_bary[i].copy()
        else:
            # Perturb two points
            i, j = rng.choice(N, size=2, replace=False)
            new_bary[i] += rng.normal(0, 0.05, 2)
            new_bary[j] += rng.normal(0, 0.05, 2)
            
        # Clip to triangle
        new_bary = _clip_bary(new_bary)
        
        new_obj = _objective(new_bary)
        
        # Acceptance probability
        delta = new_obj - current_obj
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_bary = new_bary
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_bary = current_bary.copy()
                
        T *= cooling_rate
        
    return best_bary, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_obj = float('inf')
    
    # Run multiple restarts
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        # Mix of random and structured
        if i % 5 == 0:
            bary0 = _generate_structured_bary()
        else:
            bary0 = _generate_initial_bary(rng)
            
        # Allocate time per restart
        # Decreasing time budget for later restarts? Or fixed?
        # Let's use a fixed small budget per restart to allow more restarts.
        # 140s / 20 = 7s per restart.
        t_budget = 7.0
        
        # Ensure we don't exceed deadline
        if time.time() + t_budget > deadline:
            t_budget = deadline - time.time() - 1.0
            if t_budget < 1.0:
                break
                
        bary_res, obj_res = _simulated_annealing(bary0, t_budget, rng)
        
        if obj_res < best_obj:
            best_obj = obj_res
            best_bary = bary_res
            
    if best_bary is None:
        # Fallback
        best_bary = _generate_initial_bary(rng)
        
    # Convert to Cartesian
    points = _get_cartesian(best_bary)
    
    return points

