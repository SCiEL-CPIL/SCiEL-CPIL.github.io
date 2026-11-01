import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_verts_vec(centers, angles_deg, side=1.0):
    """
    Vectorized computation of vertices for multiple hexagons.
    centers: (N, 2) array
    angles_deg: (N,) array
    Returns: (N, 6, 2) array
    """
    angles_rad = np.radians(angles_deg)
    # Angles of vertices relative to hexagon center
    # 0, 60, 120, 180, 240, 300 degrees
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    
    # (N, 1, 1) + (1, 6, 1)
    total_angles = angles_rad[:, None] + v_angles[None, :]
    
    cos_a = np.cos(total_angles)
    sin_a = np.sin(total_angles)
    
    # (N, 6, 1) * side
    dx = side * cos_a
    dy = side * sin_a
    
    # Add centers
    # centers is (N, 2) -> (N, 1, 2)
    verts = np.stack([dx + centers[:, 0:1], dy + centers[:, 1:2]], axis=-1)
    return verts

def min_outer_side_vec(inner_list, oc, oa_deg):
    """
    Vectorized minimum outer side calculation.
    inner_list: list of [x, y, angle_deg] or (N,3) array
    oc: (cx, cy)
    oa_deg: float
    """
    if isinstance(inner_list, list):
        inner_arr = np.array(inner_list)
    else:
        inner_arr = inner_list
        
    if inner_arr.size == 0:
        return 0.0

    centers = inner_arr[:, 0:2]
    angles = inner_arr[:, 2]
    
    verts = hex_verts_vec(centers, angles)
    
    # Flatten verts to (N*6, 2)
    verts_flat = verts.reshape(-1, 2)
    
    # Translate to outer center
    ox, oy = oc
    verts_rel = verts_flat - np.array([ox, oy])
    
    # Rotate to outer frame
    oa_rad = math.radians(oa_deg)
    cos_a = math.cos(-oa_rad)
    sin_a = math.sin(-oa_rad)
    
    # Rotation matrix R = [[cos, -sin], [sin, cos]]
    # v' = R v
    verts_rot = np.zeros_like(verts_rel)
    verts_rot[:, 0] = verts_rel[:, 0] * cos_a - verts_rel[:, 1] * sin_a
    verts_rot[:, 1] = verts_rel[:, 0] * sin_a + verts_rel[:, 1] * cos_a
    
    # Normals of outer hexagon
    # 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    
    # Project all vertices onto all normals
    # verts_rot: (M, 2), normals: (6, 2)
    # projs: (M, 6)
    projs = verts_rot @ normals.T
    
    # Max projection
    max_proj = np.max(projs)
    
    # Side = 2 * Apothem / sqrt(3)
    return 2.0 * max_proj / SQRT3

def smooth_outer_side_vec(inner_list, oc, oa_deg, beta=50.0):
    """
    Smooth approximation of the outer side length using log-sum-exp for the max projection.
    This provides gradients for optimization.
    """
    if isinstance(inner_list, list):
        inner_arr = np.array(inner_list)
    else:
        inner_arr = inner_list
        
    if inner_arr.size == 0:
        return 0.0

    centers = inner_arr[:, 0:2]
    angles = inner_arr[:, 2]
    
    verts = hex_verts_vec(centers, angles)
    verts_flat = verts.reshape(-1, 2)
    
    ox, oy = oc
    verts_rel = verts_flat - np.array([ox, oy])
    
    oa_rad = math.radians(oa_deg)
    cos_a = math.cos(-oa_rad)
    sin_a = math.sin(-oa_rad)
    
    verts_rot = np.zeros_like(verts_rel)
    verts_rot[:, 0] = verts_rel[:, 0] * cos_a - verts_rel[:, 1] * sin_a
    verts_rot[:, 1] = verts_rel[:, 0] * sin_a + verts_rel[:, 1] * cos_a
    
    normal_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    
    projs = verts_rot @ normals.T
    
    # Log-sum-exp for smooth max
    max_proj = np.max(projs)
    # Avoid overflow/underflow
    scaled = beta * (projs - max_proj)
    sum_exp = np.sum(np.exp(scaled))
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def overlap_penalty_vec(inner_list, min_dist=SQRT3 - 0.001):
    """
    Vectorized overlap penalty based on center distance.
    """
    if isinstance(inner_list, list):
        inner_arr = np.array(inner_list)
    else:
        inner_arr = inner_list
        
    n = inner_arr.shape[0]
    if n < 2:
        return 0.0
        
    xs = inner_arr[:, 0]
    ys = inner_arr[:, 1]
    
    # Pairwise distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    
    violations = np.maximum(0.0, min_dist - dists)
    return float(np.sum(violations * violations))

def check_disjoint(inner_list, eps=1e-7):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner_list)
    for i in range(n):
        vi = hex_verts(inner_list[i][0], inner_list[i][1], 1.0, math.radians(inner_list[i][2]))
        for j in range(i + 1, n):
            vj = hex_verts(inner_list[j][0], inner_list[j][1], 1.0, math.radians(inner_list[j][2]))
            if sat_overlap(vi, vj, eps):
                return False
    return True

def hex_verts(cx, cy, s, ang_rad):
    """Compute vertices of a regular hexagon."""
    v = np.zeros((6, 2))
    for i in range(6):
        theta = ang_rad + i * math.pi / 3.0
        v[i, 0] = cx + s * math.cos(theta)
        v[i, 1] = cy + s * math.sin(theta)
    return v

def sat_overlap(p1, p2, eps=1e-9):
    """
    Separating Axis Theorem check for two convex polygons.
    Returns True if they overlap.
    """
    axes = []
    for i in range(len(p1)):
        p_a = p1[i]
        p_b = p1[(i + 1) % len(p1)]
        edge = (p_b[0] - p_a[0], p_b[1] - p_a[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
            
    for i in range(len(p2)):
        p_a = p2[i]
        p_b = p2[(i + 1) % len(p2)]
        edge = (p_b[0] - p_a[0], p_b[1] - p_a[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
            
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in p1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in p1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in p2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in p2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135.0 # Leave 15s margin for safety
    
    # 1. Initialize with a compact hexagonal lattice sorted by distance
    coords = []
    coords.append((0,0))
    # Ring 1
    for q in range(-1, 2):
        for r in range(-1, 2):
            if q == 0 and r == 0: continue
            if max(abs(q), abs(r), abs(-q-r)) == 1:
                coords.append((q,r))
    # Ring 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            if q == 0 and r == 0: continue
            if max(abs(q), abs(r), abs(-q-r)) == 2:
                coords.append((q,r))
                
    def dist_key(c):
        q, r = c
        return 3 * (q*q + q*r + r*r)
        
    coords.sort(key=dist_key)
    axial_coords = coords[:N]
    
    scale = 1.0 + 1e-4 
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    initial_inner = np.array([[x, y, 0.0] for x, y in centers])
    
    cx_avg = np.mean(initial_inner[:, 0])
    cy_avg = np.mean(initial_inner[:, 1])
    init_outer_center = (cx_avg, cy_avg)
    init_outer_angle = 0.0
    
    best_side = float('inf')
    best_inner = initial_inner.copy()
    best_outer_center = init_outer_center
    best_outer_angle = init_outer_angle
    
    def update_best(inner, oc, oa):
        nonlocal best_side, best_inner, best_outer_center, best_outer_angle
        s = min_outer_side_vec(inner, oc, oa)
        if s < best_side and check_disjoint(inner):
            best_side = s
            best_inner = inner.copy()
            best_outer_center = (oc[0], oc[1])
            best_outer_angle = oa
            
    update_best(initial_inner, init_outer_center, init_outer_angle)
    
    # 2. Phase 1: Optimize Outer Container with Fixed Inner Layout
    def obj_outer(params):
        cx, cy, ang = params
        return min_outer_side_vec(initial_inner, (cx, cy), ang)
        
    best_outer_params = None
    best_outer_val = float('inf')
    
    init_angles = [0, 30, 60, 90, 15, 45, 75, -15, -45, -75]
    
    for ang in init_angles:
        if time.time() > deadline - 60:
            break
            
        x0 = [cx_avg, cy_avg, float(ang)]
        try:
            res = minimize(obj_outer, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_outer_val:
                best_outer_val = res.fun
                best_outer_params = res.x
        except Exception:
            pass
            
    if best_outer_params is not None:
        best_outer_center = (best_outer_params[0], best_outer_params[1])
        best_outer_angle = best_outer_params[2]
        best_side = best_outer_val
        
    # 3. Phase 2: Joint Optimization via Powell (Deterministic Local Search)
    # Use the SMOOTH objective for Powell to get better gradients and convergence
    def joint_objective_smooth(params):
        inner_list = params[:N*3].reshape(N, 3)
        ocx, ocy, oang = params[N*3], params[N*3+1], params[N*3+2]
        
        side = smooth_outer_side_vec(inner_list, (ocx, ocy), oang, beta=100.0)
        pen = overlap_penalty_vec(inner_list)
        return side + 1000.0 * pen
        
    def joint_objective_exact(params):
        inner_list = params[:N*3].reshape(N, 3)
        ocx, ocy, oang = params[N*3], params[N*3+1], params[N*3+2]
        
        side = min_outer_side_vec(inner_list, (ocx, ocy), oang)
        pen = overlap_penalty_vec(inner_list)
        return side + 1000.0 * pen

    init_params = np.concatenate([initial_inner.flatten(), [best_outer_center[0], best_outer_center[1], best_outer_angle]])
    
    starts = [
        init_params,
        init_params + np.random.normal(0, 0.1, init_params.shape),
        init_params + np.random.normal(0, 0.5, init_params.shape),
    ]
    
    if time.time() < deadline - 40:
        for _ in range(5):
            random_inner = initial_inner + np.random.normal(0, 1.0, initial_inner.shape)
            random_params = np.concatenate([random_inner.flatten(), [best_outer_center[0], best_outer_center[1], best_outer_angle]])
            starts.append(random_params)

    for s in starts:
        if time.time() > deadline - 10:
            break
        try:
            # First pass with smooth objective for better convergence
            res = minimize(joint_objective_smooth, s, method='Powell', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            res_inner = res.x[:N*3].reshape(N, 3)
            res_oc = (res.x[N*3], res.x[N*3+1])
            res_oa = res.x[N*3+2]
            
            if check_disjoint(res_inner):
                res_side = min_outer_side_vec(res_inner, res_oc, res_oa)
                if res_side < best_side:
                    best_side = res_side
                    best_inner = res_inner
                    best_outer_center = res_oc
                    best_outer_angle = res_oa
                    
            # Second pass with exact objective from the smooth result
            if time.time() < deadline - 5:
                res2 = minimize(joint_objective_exact, res.x, method='Powell', 
                               options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
                res_inner2 = res2.x[:N*3].reshape(N, 3)
                res_oc2 = (res2.x[N*3], res2.x[N*3+1])
                res_oa2 = res2.x[N*3+2]
                
                if check_disjoint(res_inner2):
                    res_side2 = min_outer_side_vec(res_inner2, res_oc2, res_oa2)
                    if res_side2 < best_side:
                        best_side = res_side2
                        best_inner = res_inner2
                        best_outer_center = res_oc2
                        best_outer_angle = res_oa2
        except Exception:
            pass
            
    # 4. Phase 3: Differential Evolution Global Search
    def de_objective(params):
        inner_list = params[:N*3].reshape(N, 3)
        ocx, ocy, oang = params[N*3], params[N*3+1], params[N*3+2]
        
        if not check_disjoint(inner_list):
            return 100.0 + 1000.0 * overlap_penalty_vec(inner_list)
            
        side = min_outer_side_vec(inner_list, (ocx, ocy), oang)
        return side
        
    bounds = []
    for _ in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((0, 360))
    bounds.append((-5, 5))
    bounds.append((-5, 5))
    bounds.append((0, 360))
    
    if time.time() < deadline - 20:
        try:
            de_res = differential_evolution(
                de_objective, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=1e-5, 
                seed=42,
                init='sobol',
                polish=False
            )
            
            if de_res.success or de_res.fun < best_side:
                res_inner = de_res.x[:N*3].reshape(N, 3)
                res_oc = (de_res.x[N*3], de_res.x[N*3+1])
                res_oa = de_res.x[N*3+2]
                
                if check_disjoint(res_inner):
                    res_side = min_outer_side_vec(res_inner, res_oc, res_oa)
                    if res_side < best_side:
                        best_side = res_side
                        best_inner = res_inner
                        best_outer_center = res_oc
                        best_outer_angle = res_oa
        except Exception:
            pass

    # 5. Phase 4: Final High-Precision Polish on Outer Container
    def obj_outer_polish(params):
        cx, cy, ang = params
        return min_outer_side_vec(best_inner, (cx, cy), ang)
        
    best_polish_val = best_side
    best_polish_params = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    starts_polish = [
        best_polish_params,
        best_polish_params + np.array([0.01, 0, 0]),
        best_polish_params + np.array([0, 0.01, 0]),
        best_polish_params + np.array([0, 0, 1.0]),
        best_polish_params + np.array([0, 0, -1.0])
    ]
    
    for s in starts_polish:
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(obj_outer_polish, s, method='Powell', 
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
            
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
        
    # 6. Phase 5: Final Joint Polish with Exact Objective
    # Do one more round of joint optimization from the best found so far
    if time.time() < deadline - 3:
        final_params = np.concatenate([best_inner.flatten(), [best_outer_center[0], best_outer_center[1], best_outer_angle]])
        
        # Try a few small perturbations
        for noise_scale in [0.05, 0.01, 0.001]:
            if time.time() > deadline - 2:
                break
            perturbed = final_params + np.random.normal(0, noise_scale, final_params.shape)
            try:
                res = minimize(joint_objective_exact, perturbed, method='Powell', 
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                
                res_inner = res.x[:N*3].reshape(N, 3)
                res_oc = (res.x[N*3], res.x[N*3+1])
                res_oa = res.x[N*3+2]
                
                if check_disjoint(res_inner):
                    res_side = min_outer_side_vec(res_inner, res_oc, res_oa)
                    if res_side < best_side:
                        best_side = res_side
                        best_inner = res_inner
                        best_outer_center = res_oc
                        best_outer_angle = res_oa
            except Exception:
                pass
                
        # Final outer polish after joint polish
        if time.time() < deadline - 1:
            final_polish_params = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
            try:
                res = minimize(obj_outer_polish, final_polish_params, method='Powell', 
                               options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res.fun < best_side:
                    best_side = res.fun
                    best_outer_center = (res.x[0], res.x[1])
                    best_outer_angle = res.x[2]
            except Exception:
                pass

    return best_inner.tolist(), best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
