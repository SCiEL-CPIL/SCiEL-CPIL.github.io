"""Combined program: hexagon packing with LP-based outer optimization and joint refinement."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

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
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
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
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap (including touching).
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

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.01):
    """
    Compute a smooth penalty for overlaps based on center distances.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    iu = np.triu_indices(n, k=1)
    dists_sq_upper = dists_sq[iu]
    
    dists = np.sqrt(dists_sq_upper)
    diff = np.maximum(0.0, min_dist - dists)
    penalty = np.sum(diff * diff)
    
    return float(penalty)

def axial_to_cartesian(axial_list, scale=1.0):
    """Convert axial coordinates to cartesian with given scale."""
    pts = []
    for q, r in axial_list:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        pts.append((x, y))
    return pts

def calc_S_for_beta(pts, beta):
    """
    Use LP to find the minimum circumradius D for a set of points
    given outer hexagon rotation beta.
    Returns (S, center, D) or (inf, None, None).
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf'), None, None
    
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy), D
        else:
            return float('inf'), None, None
    except Exception:
        return float('inf'), None, None

def optimize_outer_for_inner(inner):
    """
    Optimize outer hexagon (center, angle) to minimize side length.
    inner: list of (x, y, angle_deg)
    Returns (side, center, angle_deg)
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback
        return compute_required_side(inner, (0, 0), 0), (0, 0), 0.0
    
    # Compute cartesian vertices for all inner hexagons
    all_verts = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        all_verts.extend(verts)
    
    pts = np.array(all_verts)
    
    best_S = float('inf')
    best_beta = 0.0
    best_center = (0, 0)
    
    # Coarse search over angles
    for i in range(37):
        beta = i * (math.pi / 6) / 36
        S, center, D = calc_S_for_beta(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center
    
    # Fine search using minimize_scalar
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], 
                                 bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100, 'xatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(pts, best_beta)
    except Exception:
        pass
    
    # Convert beta to degrees
    best_angle_deg = math.degrees(best_beta)
    
    return best_S, best_center, best_angle_deg

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Define multiple candidate layouts
    candidates = []
    
    # Layout 1: Compact hexagonal flower
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-1, 2), (-2, 1)
    ]
    candidates.append(axial_1)
    
    # Layout 2: Extended ring
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)
    
    # Layout 3: Rectangular-ish
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)
    
    # Layout 4: Wide rectangle
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)
    
    # Layout 5: Tall rectangle
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)
    
    # Layout 6: Asymmetric
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_6)
    
    # Layout 7: Star pattern
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)
    
    # Layout 8: Long strip
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)
    
    # Layout 9: Diagonal
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)
    
    # Layout 10: Compact block
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try each candidate layout
    for layout_idx, axial_coords in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        # Convert to cartesian with slight scale to ensure disjointness
        epsilon = 1e-4
        scale = 1.0 + epsilon
        
        centers = axial_to_cartesian(axial_coords, scale)
        
        # Initial layout: All hexagons aligned (angle 0)
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # Optimize outer hexagon using LP
        side, center, angle_deg = optimize_outer_for_inner(initial_inner)
        
        if side < best_side:
            best_side = side
            best_inner = initial_inner
            best_outer_center = center
            best_outer_angle = angle_deg
    
    # Joint optimization to refine inner positions
    if best_inner is None:
        # Fallback to first layout
        axial_coords = candidates[0][:N]
        centers = axial_to_cartesian(axial_coords, 1.0 + 1e-4)
        best_inner = [[x, y, 0.0] for x, y in centers]
        side, center, angle_deg = optimize_outer_for_inner(best_inner)
        best_side = side
        best_outer_center = center
        best_outer_angle = angle_deg
    
    # Joint optimization with smooth surrogate
    def objective_joint_smooth(params, beta=80.0):
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
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    initial_params = []
    for x, y, a in best_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_params = np.array(initial_params)
    current_val = objective_joint_smooth(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Local search with decreasing step sizes
    step_size = 0.05
    for _ in range(1500):
        if time.time() > deadline - 25:
            break
        
        num_perturb = 4
        indices = np.random.choice(len(current_params), num_perturb, replace=False)
        
        neighbor = current_params.copy()
        for idx in indices:
            if idx % 3 == 2:
                delta = np.random.normal(0, step_size * 0.5)
            else:
                delta = np.random.normal(0, step_size)
            neighbor[idx] += delta
            
        neighbor_val = objective_joint_smooth(neighbor)
        
        if neighbor_val < current_val:
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
        else:
            if np.random.random() < 0.05:
                current_params = neighbor
                current_val = neighbor_val
        
        step_size *= 0.995
    
    best_inner_joint = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_joint.append([x, y, a])
    
    best_ocx = best_params[N*3]
    best_ocy = best_params[N*3+1]
    best_oangle = best_params[N*3+2]
    
    if check_all_disjoint(best_inner_joint, eps=1e-5):
        best_inner = best_inner_joint
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        pass

    # Final polish using LP-based optimization
    if time.time() < deadline - 10:
        side, center, angle_deg = optimize_outer_for_inner(best_inner)
        if side < best_side:
            best_side = side
            best_outer_center = center
            best_outer_angle = angle_deg
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
