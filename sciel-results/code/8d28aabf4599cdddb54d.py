"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices_vec(centers, angles_rad, side=1.0):
    """
    Vectorized computation of hexagon vertices.
    centers: (N, 2) array
    angles_rad: (N,) array
    Returns: (N, 6, 2) array of vertices
    """
    N = len(centers)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    all_angles = angles_rad[:, np.newaxis] + base_angles[np.newaxis, :]
    
    cos_a = np.cos(all_angles) 
    sin_a = np.sin(all_angles)
    
    offsets = np.stack([side * cos_a, side * sin_a], axis=-1)
    centers_expanded = centers[:, np.newaxis, :]
    
    return centers_expanded + offsets

def compute_required_side_exact(inner, outer_center, outer_angle_deg):
    """
    Exact computation of the minimum outer side length to contain all inner hexagons.
    """
    if not inner:
        return 0.0
    
    inner_arr = np.array(inner)
    centers = inner_arr[:, :2]
    angles_deg = inner_arr[:, 2]
    angles_rad = np.deg2rad(angles_deg)
    
    verts = get_hex_vertices_vec(centers, angles_rad, 1.0)
    all_verts = verts.reshape(-1, 2)
    
    ox, oy = outer_center
    all_verts = all_verts - np.array([ox, oy])
    
    theta = np.deg2rad(outer_angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    x = all_verts[:, 0]
    y = all_verts[:, 1]
    
    lx = x * cos_t + y * sin_t
    ly = -x * sin_t + y * cos_t
    
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    
    projections = all_verts.dot(normals.T)
    max_projections = np.max(projections, axis=0)
    max_apothem = np.max(max_projections)
    
    if max_apothem < 0:
        max_apothem = 0.0
        
    side = 2.0 * max_apothem / SQRT3
    return float(side)

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth approximation of the required side length using log-sum-exp for max.
    """
    if not inner:
        return 0.0
    
    inner_arr = np.array(inner)
    centers = inner_arr[:, :2]
    angles_deg = inner_arr[:, 2]
    angles_rad = np.deg2rad(angles_deg)
    
    verts = get_hex_vertices_vec(centers, angles_rad, 1.0)
    all_verts = verts.reshape(-1, 2)
    
    ox, oy = outer_center
    all_verts = all_verts - np.array([ox, oy])
    
    theta = np.deg2rad(outer_angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    x = all_verts[:, 0]
    y = all_verts[:, 1]
    
    lx = x * cos_t + y * sin_t
    ly = -x * sin_t + y * cos_t
    
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    
    # Projections: (M, 6)
    projections = all_verts.dot(normals.T)
    
    # Soft max over vertices for each normal
    # logsumexp(projections / beta) * beta
    # We want max over all projections (vertices and normals)
    # This is equivalent to max over k of (max over v of proj[v,k])
    # We can flatten to 1D and do a single logsumexp
    flat_projections = projections.flatten()
    
    # To avoid overflow, subtract max
    max_val = np.max(flat_projections)
    if max_val > 100: # Safety
        max_val = 100
    shifted = flat_projections - max_val
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = max_val + np.log(len(flat_projections)) + log_sum - np.log(len(flat_projections)) # This is just max + log(sum(exp))
    # Actually: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    # So soft_max = max_val + log(sum(exp(shifted)))
    # The standard soft max is (1/beta) * logsumexp(x/beta). 
    # Here we just want a smooth max.
    smooth_max = max_val + np.log(np.sum(np.exp(shifted)))
    
    # The beta parameter controls sharpness. 
    # If we divide projections by beta before logsumexp:
    # smooth_max_beta = (1/beta) * (max(flat/beta) + log(sum(exp(flat/beta - max(flat/beta)))))
    # Let's stick to a simple smooth max with a fixed scale
    scale = 20.0
    scaled = flat_projections / scale
    max_scaled = np.max(scaled)
    smooth_scaled = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    max_apothem = smooth_scaled * scale
    
    if max_apothem < 0:
        max_apothem = 0.0
        
    side = 2.0 * max_apothem / SQRT3
    return float(side)

def compute_overlap_penalty_fast(inner, min_dist=SQRT3 - 0.001):
    """
    Fast computation of overlap penalty based on center distances.
    """
    inner_arr = np.array(inner)
    centers = inner_arr[:, :2]
    N = len(centers)
    
    if N < 2:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    
    diffs = min_dist - pair_dists
    positive_diffs = np.maximum(diffs, 0)
    penalty = np.sum(positive_diffs**2)
    
    return float(penalty)

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def generate_initial_layouts():
    """
    Generate multiple high-quality initial layouts for N=11 hexagons.
    """
    layouts = []
    
    # Layout 1: Standard hexagonal cluster (1 + 6 + 4)
    # Center at (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 4 hexagons placed to fill gaps
    
    # Axial coordinates for a compact 11-hexagon shape
    # One known compact shape:
    #   . X .
    #  X X X X
    #   X X X
    #  . X .
    # This is 1+4+4+1 = 10? No.
    # Let's use a known packing:
    # A 3x3 grid of hexagons has 9. Add 2 more.
    # Or a "flower" of 7, plus 4 more around the perimeter.
    
    # Using Cartesian coords for simplicity, then we'll optimize
    # Scale factor 1.0 + epsilon to start slightly separated
    eps = 1e-4
    s = 1.0 + eps
    
    # Layout 1: 1 center, 6 neighbors, 4 more
    centers = [(0, 0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = 1.5 * s * math.cos(angle)
        y = SQRT3 * s * math.sin(angle)
        centers.append((x, y))
    
    # Add 4 more at the "corners" of the hexagon
    # The outer vertices of the 7-hex cluster are at distance 2*1.5 = 3 from center?
    # No, the apothem of the 7-hex cluster is 1.5*SQRT3 + 1 = 3.598?
    # Let's just place them at the midpoints of the outer edges of the 7-hex cluster
    # Outer vertices of the 7-hex cluster are at angles 0, 60, ...
    # The "gaps" are at angles 30, 90, ...
    # Distance from center to gap:
    # The 7-hex cluster has a bounding hexagon of side length 2 (in terms of unit hexagons).
    # The "corners" of this bounding hexagon are at distance 2 * 1.5 * SQRT3? No.
    # Let's just try placing them at specific offsets.
    
    # Better: Use a known good packing for 11.
    # Reference: "Packing unit hexagons in a hexagon"
    # For N=11, the optimal packing is likely a distorted lattice.
    
    # Let's generate a grid of candidate points and pick 11
    # Or just use a few heuristic layouts.
    
    # Layout 2: Two rows of 5 and 6?
    # Row 1: 5 hexagons
    # Row 2: 6 hexagons
    # This forms a parallelogram shape.
    row1 = []
    row2 = []
    for i in range(5):
        x = 1.5 * s * i
        y = 0
        row1.append((x, y))
    for i in range(6):
        x = 1.5 * s * (i - 0.5)
        y = SQRT3 * s
        row2.append((x, y))
    centers_2 = row1 + row2
    layouts.append(centers_2)
    
    # Layout 3: Hexagonal lattice 1-6-4
    # Center (0,0)
    # 6 neighbors
    # 4 more at (3, 0), (-3, 0), (1.5, 2.6), (-1.5, -2.6) approx
    centers_3 = [(0, 0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = 1.5 * s * math.cos(angle)
        y = SQRT3 * s * math.sin(angle)
        centers_3.append((x, y))
    # Add 4 more
    # Place them at the 4 "diagonal" positions of the 7-hex cluster
    # The 7-hex cluster has vertices at (3,0), (1.5, 2.6), (-1.5, 2.6), (-3,0), (-1.5,-2.6), (1.5,-2.6)
    # Wait, the outer hexagons in the 7-cluster are centered at distance 1.5*SQRT3? No.
    # Center to neighbor center distance is 2 * apothem = SQRT3 * 1 = 1.732?
    # No, side length 1 hexagon has apothem SQRT3/2.
    # Distance between centers of touching hexagons is 2 * SQRT3/2 = SQRT3.
    # So neighbors are at distance SQRT3.
    # My previous code used 1.5 * s. 1.5 is the distance for side length 1 hexagons in a lattice?
    # Yes, x = 1.5 * q. If q=1, x=1.5.
    # y = SQRT3 * (r + q/2). If r=0, q=1, y=SQRT3/2.
    # Dist = sqrt(1.5^2 + (SQRT3/2)^2) = sqrt(2.25 + 0.75) = sqrt(3) = SQRT3. Correct.
    
    # So for Layout 3, we have 7 hexagons.
    # We need 4 more.
    # Place them at (3, 0), (-3, 0), (1.5, 2.6), (-1.5, -2.6) ?
    # (3,0) is 2 steps away in q.
    # Let's add (3,0), (-3,0), (1.5, 2.6), (-1.5, -2.6)
    centers_3.append((3.0 * s, 0))
    centers_3.append((-3.0 * s, 0))
    centers_3.append((1.5 * s, SQRT3 * s * 1.5))
    centers_3.append((-1.5 * s, -SQRT3 * s * 1.5))
    layouts.append(centers_3)
    
    # Layout 4: Random perturbation of Layout 1
    centers_4 = [c + (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)) for c in centers_3]
    layouts.append(centers_4)
    
    return layouts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_side_global = float('inf')
    best_inner_global = None
    best_outer_center_global = (0.0, 0.0)
    best_outer_angle_global = 0.0
    
    layouts = generate_initial_layouts()
    
    for layout_centers in layouts:
        if time.time() > deadline - 30:
            break
            
        # Initial inner: centers from layout, angle 0
        initial_inner = [[x, y, 0.0] for x, y in layout_centers]
        
        # Optimize outer first to center the cluster
        def objective_outer_fast(params):
            cx, cy, angle_deg = params
            return compute_required_side_exact(initial_inner, (cx, cy), angle_deg)
            
        cx_avg = sum(c[0] for c in layout_centers) / len(layout_centers)
        cy_avg = sum(c[1] for c in layout_centers) / len(layout_centers)
        
        best_outer_val = float('inf')
        best_outer_params = None
        
        # Try a few starting angles
        for init_angle in [0, 15, 30, 45, 60]:
            if time.time() > deadline - 20:
                break
            init_params = np.array([cx_avg, cy_avg, float(init_angle)])
            try:
                res = minimize(objective_outer_fast, init_params, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_outer_val:
                    best_outer_val = res.fun
                    best_outer_params = res.x
            except Exception:
                pass
        
        if best_outer_params is None:
            best_outer_params = np.array([cx_avg, cy_avg, 0.0])
            best_outer_val = objective_outer_fast(best_outer_params)
            
        best_outer_center = (best_outer_params[0], best_outer_params[1])
        best_outer_angle = best_outer_params[2]
        
        # Joint Optimization
        # Parameters: x1, y1, a1, ..., xN, yN, aN, cx, cy, cangle
        def objective_joint(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]
            
            side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle)
            penalty = compute_overlap_penalty_fast(inner_list)
            
            return side + 100.0 * penalty
        
        # Initial parameters
        init_params = []
        for x, y, a in initial_inner:
            init_params.extend([x, y, a])
        init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
        init_params = np.array(init_params)
        
        best_joint_val = objective_joint(init_params)
        best_joint_params = init_params.copy()
        
        # Multi-start local search
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Perturb
            noise_scale = 0.5 if restart == 0 else 0.2
            current_params = init_params + np.random.normal(0, noise_scale, init_params.shape)
            # Keep angles in range
            current_params[2::3][:N] = np.mod(current_params[2::3][:N], 360)
            current_params[N*3+2] = np.mod(current_params[N*3+2], 360)
            
            current_val = objective_joint(current_params)
            
            try:
                res = minimize(objective_joint, current_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_joint_val:
                    best_joint_val = res.fun
                    best_joint_params = res.x
            except Exception:
                pass
            
            # Polish
            if time.time() > deadline - 5:
                break
                
            for _ in range(3):
                if time.time() > deadline - 2:
                    break
                shake_params = best_joint_params + np.random.normal(0, 0.1, best_joint_params.shape)
                shake_params[2::3][:N] = np.mod(shake_params[2::3][:N], 360)
                shake_params[N*3+2] = np.mod(shake_params[N*3+2], 360)
                
                try:
                    res_shake = minimize(objective_joint, shake_params, method='Nelder-Mead',
                                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res_shake.fun < best_joint_val:
                        best_joint_val = res_shake.fun
                        best_joint_params = res_shake.x
                except Exception:
                    pass
        
        # Extract best
        best_inner_local = []
        for i in range(N):
            x = best_joint_params[i*3]
            y = best_joint_params[i*3+1]
            a = best_joint_params[i*3+2]
            best_inner_local.append([x, y, a])
            
        best_ocx = best_joint_params[N*3]
        best_ocy = best_joint_params[N*3+1]
        best_oangle = best_joint_params[N*3+2]
        
        # Check validity
        if check_all_disjoint(best_inner_local, eps=1e-5):
            # Final polish of outer only
            def objective_outer_polish(params):
                cx, cy, angle_deg = params
                return compute_required_side_exact(best_inner_local, (cx, cy), angle_deg)
                
            try:
                res_final = minimize(objective_outer_polish, [best_ocx, best_ocy, best_oangle], 
                                     method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                if res_final.fun < best_side_global:
                    best_side_global = res_final.fun
                    best_inner_global = best_inner_local
                    best_outer_center_global = (res_final.x[0], res_final.x[1])
                    best_outer_angle_global = res_final.x[2]
            except Exception:
                pass
            
            if best_inner_global is None:
                best_side_global = compute_required_side_exact(best_inner_local, (best_ocx, best_ocy), best_oangle)
                best_inner_global = best_inner_local
                best_outer_center_global = (best_ocx, best_ocy)
                best_outer_angle_global = best_oangle

    # Fallback
    if best_inner_global is None:
        best_inner_global = initial_inner
        best_outer_center_global = (0.0, 0.0)
        best_outer_angle_global = 0.0
        best_side_global = compute_required_side_exact(best_inner_global, best_outer_center_global, best_outer_angle_global)

    return best_inner_global, best_outer_center_global, best_side_global, best_outer_angle_global

# EVOLVE-BLOCK-END
