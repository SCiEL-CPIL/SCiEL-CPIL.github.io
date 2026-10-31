"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    all_projections = []
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                all_projections.append(proj)
    
    if not all_projections:
        return 0.0
    
    # Soft max using log-sum-exp
    max_proj = max(all_projections)
    if max_proj < -1000:
        return 0.0
    exp_sum = 0.0
    for p in all_projections:
        exp_sum += math.exp(beta * (p - max_proj))
    soft_max = max_proj + math.log(exp_sum) / beta
    
    return 2.0 * soft_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
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
        
        # Check for gap
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

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.01):
    """
    Compute a smooth penalty for overlaps based on center distances.
    This is a proxy for actual overlap, valid for aligned hexagons.
    Vectorized for speed.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    # Extract positions
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    # Compute all pairwise squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    # Only consider i < j pairs (upper triangle)
    iu = np.triu_indices(n, k=1)
    dists_sq_upper = dists_sq[iu]
    
    # Penalty: sum of (min_dist - dist)^2 where dist < min_dist
    dists = np.sqrt(dists_sq_upper)
    diff = np.maximum(0.0, min_dist - dists)
    penalty = np.sum(diff * diff)
    
    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Define 11 hexagons on a hexagonal lattice - more compact arrangement
    # Use a "flower" pattern: center + 6 ring + 4 more strategically placed
    # Axial coordinates (q, r) for a compact 11-hex cluster
    axial_coords = [
        (0, 0),      # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # 6 neighbors
        (2, -1), (1, 0), (0, 1), (-1, 1)  # 4 more - but this duplicates
    ]
    
    # Better: use a 3-4-3 arrangement or similar compact pattern
    # Let's use: center, 6 around, then 4 in the "gaps" of the second ring
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-1, 2), (-2, 1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian with tight spacing
    # Standard hex grid: 
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Use slightly larger epsilon to give optimizer room to tighten
    epsilon = 0.02
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Calculate centroid
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # 3. Optimize Outer Hexagon parameters (cx, cy, angle)
    # The inner hexagons are fixed in this phase (rigid body assumption)
    
    def objective_outer(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    # The optimal angle is likely related to the symmetry of the cluster.
    # Focus on angles near 0, 30, 60 which align with hexagonal symmetry
    init_angles = [0, 10, 20, 30, 40, 50, 60, 150, 160, 170, 180, 190, 200, 210, 300, 310, 320, 330, 340, 350]
    
    for init_angle in init_angles:
        if time.time() > deadline - 50:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 12000})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        # Fallback
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # 4. Joint Optimization: Allow perturbations of inner hexagons
    # Use smooth surrogate for the outer constraint to make optimization easier
    
    def objective_joint_smooth(params, beta=80.0):
        # Unpack params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, angle]
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta)
        
        # Penalty for overlaps - use a tighter min_dist to encourage packing
        penalty = compute_overlap_penalty(inner_list, min_dist=SQRT3 - 0.005)
        
        return side + 500.0 * penalty
    
    # Initial params: current best inner + best outer
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Use Nelder-Mead on the smooth objective
    current_params = np.array(initial_params)
    current_val = objective_joint_smooth(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Local search with decreasing step sizes and adaptive beta
    step_size = 0.08
    beta = 40.0
    for _ in range(3000):
        if time.time() > deadline - 30:
            break
        
        # Randomly pick a few dimensions to perturb
        num_perturb = 5
        indices = np.random.choice(len(current_params), num_perturb, replace=False)
        
        neighbor = current_params.copy()
        for idx in indices:
            # Different step sizes for different parameter types
            if idx % 3 == 2:  # angle
                delta = np.random.normal(0, step_size * 0.3)
            else:  # position
                delta = np.random.normal(0, step_size)
            neighbor[idx] += delta
            
        neighbor_val = objective_joint_smooth(neighbor, beta)
        
        if neighbor_val < current_val:
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
        else:
            # Occasionally accept worse solutions to escape local minima
            if np.random.random() < 0.08:
                current_params = neighbor
                current_val = neighbor_val
        
        # Decay step size and increase beta for sharper approximation
        step_size *= 0.996
        beta = min(100.0, beta * 1.002)
    
    # Reconstruct best joint solution
    best_inner_joint = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_joint.append([x, y, a])
    
    best_ocx = best_params[N*3]
    best_ocy = best_params[N*3+1]
    best_oangle = best_params[N*3+2]
    
    # Verify with SAT
    if check_all_disjoint(best_inner_joint, eps=1e-5):
        best_inner = best_inner_joint
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        # Fallback to rigid body result if joint search broke disjointness
        best_inner = initial_inner
        best_side = best_val

    # 5. Final Polish: Tighten the outer container around the fixed inner layout
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    # Try more diverse local starts around the current best
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 5],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 5],
        [best_outer_center[0] + 0.05, best_outer_center[1], best_outer_angle],
        [best_outer_center[0] - 0.05, best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1] + 0.05, best_outer_angle],
        [best_outer_center[0], best_outer_center[1] - 0.05, best_outer_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 20000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    # 6. Ultra-precise final refinement with very tight tolerances
    if time.time() < deadline - 1:
        try:
            res = minimize(objective_outer_polish, best_polish_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 40000})
            if res.fun < best_side:
                best_outer_center = (res.x[0], res.x[1])
                best_outer_angle = res.x[2]
                best_side = res.fun
        except Exception:
            pass
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
