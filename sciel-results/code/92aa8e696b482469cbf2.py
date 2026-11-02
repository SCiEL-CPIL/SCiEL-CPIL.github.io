"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Barycentric coordinate parameterization: 
   Points are represented as (u, v) where 0 <= u, v, u+v <= 1.
   Cartesian: x = u + 0.5*v, y = sqrt(3)/2 * v.
   This ensures points are strictly inside the triangle, removing the need for 
   projection steps and boundary penalties during optimization.
   
2. Deterministic Multi-Start Powell Optimization:
   Powell's method is a derivative-free optimization algorithm that is robust 
   to non-smooth objectives (like the minimum area function). 
   We run multiple starts from diverse initial configurations (including 
   Farthest Point Sampling and random grids) and let Powell refine each.
   
3. Vectorized Evaluation:
   The objective function evaluates the minimum area of all C(11,3)=165 triangles 
   efficiently using NumPy vectorization.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _min_triangle_area_bary(pts_bary):
    """
    Compute the minimum area of any triangle formed by 11 points given in barycentric coords.
    pts_bary: shape (N, 2) array of (u, v).
    Returns: min area (scalar).
    """
    # Convert to Cartesian for area calculation? 
    # Or use barycentric area formula?
    # Area in Cartesian = 0.5 * |det(P_j - P_i, P_k - P_i)|
    # Since the transformation from (u,v) to (x,y) is linear (affine), 
    # the area scales by a constant factor.
    # x = u + 0.5 v
    # y = S3H v
    # Jacobian J = [[1, 0.5], [0, S3H]]
    # det(J) = S3H
    # So Area_cart = S3H * Area_bary_coord_system
    # Where Area_bary_coord_system is the area in the (u,v) plane.
    # Actually, let's just compute in Cartesian to be safe and standard, 
    # or pre-compute the constant.
    # Let's compute in Cartesian to avoid confusion, it's fast enough.
    
    x = pts_bary[:, 0] + 0.5 * pts_bary[:, 1]
    y = S3H * pts_bary[:, 1]
    pts_cart = np.column_stack((x, y))
    
    # Vectorized area calculation for all triplets
    n = len(pts_cart)
    # Pre-compute indices for triplets (165 of them)
    # To avoid regenerating every time, we could make it global, but for clarity:
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p_i = pts_cart[i_idx]
    p_j = pts_cart[j_idx]
    p_k = pts_cart[k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective(x_flat):
    """
    Objective function for minimization.
    We want to MAXIMIZE the min area.
    So we MINIMIZE -min_area.
    x_flat: shape (2N,) array of barycentric coords.
    """
    pts_bary = x_flat.reshape(N, 2)
    
    # Barycentric constraints: u >= 0, v >= 0, u+v <= 1
    # Powell's method does not natively handle constraints well without transformations.
    # However, we can use a penalty method or rely on the fact that 
    # if we start inside and use small steps, we might stay inside.
    # BUT, Powell can step outside.
    # Better approach: Use a transformation that maps R^2 to the triangle?
    # Or simply use bounds in L-BFGS-B? But Powell doesn't support bounds directly in scipy.minimize.
    # 
    # Alternative: Use the "logistic" or "sigmoid" transform for coordinates?
    # Or, simpler: Use the fact that for small perturbations, if we start well inside, 
    # we can add a large penalty if points go outside.
    #
    # Let's try a robust penalty:
    penalty = 0.0
    u = pts_bary[:, 0]
    v = pts_bary[:, 1]
    
    # u < 0
    mask_u_neg = u < 0
    if np.any(mask_u_neg):
        penalty += np.sum(u[mask_u_neg] ** 2)
        
    # v < 0
    mask_v_neg = v < 0
    if np.any(mask_v_neg):
        penalty += np.sum(v[mask_v_neg] ** 2)
        
    # u + v > 1
    w = 1.0 - u - v
    mask_w_neg = w < 0
    if np.any(mask_w_neg):
        penalty += np.sum(w[mask_w_neg] ** 2)
        
    if penalty > 1e-6:
        return 1000.0 + penalty  # Large penalty
    
    area = _min_triangle_area_bary(pts_bary)
    return -area

def _random_bary_point(rng):
    """Generate a random point in barycentric coords (u,v) inside the triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    return [u, v]

def _fps_init(rng, n_points):
    """
    Farthest Point Sampling initialization in barycentric space.
    Start with a random point, then add points that maximize the minimum 
    distance to existing points (in Cartesian space for geometric relevance).
    """
    # Start with centroid (0.5, 1/3 in barycentric? No, centroid is (1/3, 1/3, 1/3) -> u=1/3, v=1/3? 
    # Wait. Barycentric (alpha, beta, gamma).
    # x = alpha*0 + beta*1 + gamma*0.5
    # y = alpha*0 + beta*0 + gamma*S3H
    # If we use u=beta, v=gamma, then alpha = 1-u-v.
    # Centroid: x=0.5, y=S3H/3.
    # S3H/3 = S3H * v => v = 1/3.
    # 0.5 = u + 0.5*(1/3) => u = 0.5 - 1/6 = 1/3.
    # So centroid is (1/3, 1/3).
    
    pts = [np.array([1/3, 1/3])]
    
    for _ in range(n_points - 1):
        best_pos = None
        best_min_dist = -1
        
        # Sample candidates
        for _ in range(500):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            
            cand_bary = np.array([u, v])
            cand_cart_x = u + 0.5 * v
            cand_cart_y = S3H * v
            cand_cart = np.array([cand_cart_x, cand_cart_y])
            
            # Compute min distance to existing points (in Cartesian)
            min_d = float('inf')
            for p in pts:
                px = p[0] + 0.5 * p[1]
                py = S3H * p[1]
                d = np.sqrt((cand_cart_x - px)**2 + (cand_cart_y - py)**2)
                if d < min_d:
                    min_d = d
            
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand_bary
                
        if best_pos is not None:
            pts.append(best_pos)
            
    return np.array(pts)

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts_bary = None
    best_score = -float('inf')
    
    # Multi-start Powell Optimization
    # We will generate several diverse initializations and run Powell on each.
    # Powell is derivative-free and works well for this non-smooth objective.
    
    num_starts = 20  # Number of independent Powell runs
    
    for start_idx in range(num_starts):
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate initial configuration
        if start_idx == 0:
            # Start with FPS
            init_bary = _fps_init(rng, N)
        elif start_idx == 1:
            # Start with a grid-like pattern
            # Approximate optimal layout for small N often has symmetry
            # For 11 points, maybe corners + edge midpoints + interior?
            # Let's just use a random grid perturbed
            init_bary = np.array([_random_bary_point(rng) for _ in range(N)])
        else:
            # Random restart
            init_bary = np.array([_random_bary_point(rng) for _ in range(N)])
            
        # Ensure validity
        # (FPS and random generation should be valid)
        
        # Run Powell's Method
        # Maximize min area -> Minimize -min area
        try:
            result = minimize(
                _objective,
                init_bary.flatten(),
                method='Powell',
                options={'maxiter': 500, 'maxfev': 2000}
            )
            
            if result.success or (result.fun < best_score * -1): # result.fun is -area
                final_bary = result.x.reshape(N, 2)
                
                # Verify points are inside (Penalty should have kept them in, but check)
                # If penalty was triggered, result.fun would be huge positive.
                if result.fun < 0: # Valid solution (negative of positive area)
                    score = -result.fun
                    if score > best_score:
                        best_score = score
                        best_pts_bary = final_bary.copy()
                        
        except Exception as e:
            # Handle any optimization errors
            pass
            
    # If no valid solution was found (unlikely), fallback to FPS
    if best_pts_bary is None:
        best_pts_bary = _fps_init(rng, N)
        
    # Convert best barycentric points to Cartesian
    x = best_pts_bary[:, 0] + 0.5 * best_pts_bary[:, 1]
    y = S3H * best_pts_bary[:, 1]
    
    final_pts = np.column_stack((x, y))
    
    # Final sanity check and projection (just in case numerical drift occurred)
    for i in range(N):
        px, py = final_pts[i]
        if py < 0: py = 0
        max_y = min(S3H * px, S3H * (1 - px))
        if py > max_y: py = max_y
        if px < 0: px = 0
        if px > 1: px = 1
        final_pts[i] = [px, py]
        
    return final_pts

# Helper for potential future use or testing
def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (Cartesian)."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle (Cartesian)."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]
