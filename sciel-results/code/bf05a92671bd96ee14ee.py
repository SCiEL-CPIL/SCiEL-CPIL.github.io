"""
Packing 11 unit hexagons in a regular hexagon.
Strategy: Axial lattice cluster + joint optimization (positions, angles, container).
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array (6, 2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(thetas),
        cy + side * np.sin(thetas)
    ])

def hex_verts_local(side=1.0, angle_rad=0.0):
    """6 vertices of unit hexagon at origin as numpy array (6, 2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        side * np.cos(thetas),
        side * np.sin(thetas)
    ])

def container_normals(outer_angle_deg):
    """
    6 outward normals of the container hexagon.
    The container is centered at origin for normal calculation, 
    then rotated by outer_angle_deg.
    """
    base_angles = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    a = math.radians(outer_angle_deg)
    cs, sn = math.cos(a), math.sin(a)
    rot = np.array([[cs, -sn], [sn, cs]])
    return normals @ rot.T

def compute_required_side_inner(inner, outer_center, outer_angle_deg):
    """
    Compute minimum side length of outer hexagon to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (cx, cy)
    outer_angle_deg: rotation of outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        diffs = verts - np.array([ox, oy])
        projs = diffs @ normals.T
        current_max = np.max(projs)
        if current_max > max_proj:
            max_proj = current_max
            
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth approximation of compute_required_side_inner."""
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    
    projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        diffs = verts - np.array([ox, oy])
        projs.append(diffs @ normals.T)
        
    all_projs = np.concatenate(projs, axis=0).flatten()
    
    if len(all_projs) == 0:
        return 0.0
        
    max_proj = np.max(all_projs)
    if max_proj < -1000:
        return 0.0
        
    scaled = beta * (all_projs - max_proj)
    smooth_max = max_proj + math.log(np.sum(np.exp(scaled))) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_mtv(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    Returns (is_overlap, mtv_vector, mtv_magnitude)
    """
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
                
    min_overlap = float('inf')
    mtv = None
    
    for axis in axes:
        min1 = np.min(v1 @ axis)
        max1 = np.max(v1 @ axis)
        min2 = np.min(v2 @ axis)
        max2 = np.max(v2 @ axis)
        
        if max1 < min2 + eps:
            overlap = min2 - max1
            if overlap < min_overlap:
                min_overlap = overlap
                mtv = axis * overlap
        elif max2 < min1 + eps:
            overlap = min1 - max2
            if overlap < min_overlap:
                min_overlap = overlap
                mtv = -axis * overlap
        else:
            return (True, None, 0.0)
            
    if mtv is not None:
        return (False, mtv, min_overlap)
    return (True, None, 0.0)

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    polys = [get_hex_vertices(p[0], p[1], 1.0, math.radians(p[2])) for p in inner]
    for i in range(n):
        for j in range(i + 1, n):
            overlap, _, _ = sat_overlap_mtv(polys[i], polys[j], eps)
            if overlap:
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Smooth penalty for overlaps based on center distances."""
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            dist = math.hypot(xi - xj, yi - yj)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
    return penalty

def project_disjoint(inner, min_dist=SQRT3 - 0.001):
    """
    Project overlapping hexagons apart by adjusting positions.
    Returns a new list of (x, y, angle) with reduced overlaps.
    """
    n = len(inner)
    new_inner = [list(p) for p in inner]
    
    for iteration in range(20):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ai = new_inner[i]
                xj, yj, aj = new_inner[j]
                dx = xj - xi
                dy = yj - yi
                dist = math.hypot(dx, dy)
                
                if dist < min_dist:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2*math.pi)
                        dx = math.cos(angle)
                        dy = math.sin(angle)
                        dist = 1.0
                    
                    push = (min_dist - dist) / 2.0 + 0.001
                    ux = dx / dist
                    uy = dy / dist
                    
                    new_inner[i][0] -= push * ux
                    new_inner[i][1] -= push * uy
                    new_inner[j][0] += push * ux
                    new_inner[j][1] += push * uy
                    moved = True
        
        if not moved:
            break
    
    return [tuple(p) for p in new_inner]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Hexagonal lattice initialization
    # 11 hexagons: 1 center, 6 ring 1, 4 from ring 2
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    epsilon = 1e-6
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # Phase 1: Smooth optimization of outer container
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
    
    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        return compute_required_side(initial_inner, (cx, cy), angle_deg)
    
    best_result = None
    best_val = float('inf')
    
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    for init_angle in init_angles:
        if time.time() > deadline - 50:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res_smooth = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                                  options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 600})
            res_exact = minimize(objective_outer_exact, res_smooth.x, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
            
            if res_exact.fun < best_val:
                best_val = res_exact.fun
                best_result = res_exact.x
        except Exception:
            pass
    
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        best_side = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0

    # Phase 2: Joint optimization
    def objective_joint_smooth(params, beta=60.0):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta=beta)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 500.0 * penalty
    
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_params = np.array(initial_params)
    current_val = objective_joint_smooth(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    T = 1.5
    T_min = 0.005
    cooling = 0.9995
    
    max_iter = 10000
    for i in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        neighbor = current_params.copy()
        
        if np.random.random() < 0.75:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            if dim == 2:
                delta = np.random.normal(0, 3.0)
            else:
                delta = np.random.normal(0, 0.25)
            neighbor[idx*3 + dim] += delta
        else:
            dim = np.random.randint(N*3, N*3+3)
            if dim == N*3+2:
                delta = np.random.normal(0, 4.0)
            else:
                delta = np.random.normal(0, 0.15)
            neighbor[dim] += delta
        
        inner_list = []
        for k in range(N):
            inner_list.append([neighbor[k*3], neighbor[k*3+1], neighbor[k*3+2]])
        
        projected_inner = project_disjoint(inner_list)
        
        for k in range(N):
            neighbor[k*3] = projected_inner[k][0]
            neighbor[k*3+1] = projected_inner[k][1]
            neighbor[k*3+2] = projected_inner[k][2]
        
        neighbor_val = objective_joint_smooth(neighbor)
        
        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, 1e-10)):
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
        
        T *= cooling
        if T < T_min:
            break
    
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
        fixed_inner = project_disjoint(best_inner_joint)
        if check_all_disjoint(fixed_inner, eps=1e-5):
            best_inner = fixed_inner
            best_outer_center = (best_ocx, best_ocy)
            best_outer_angle = best_oangle
            best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
        else:
            best_inner = initial_inner
            best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # Phase 3: Final high-precision polish
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 3],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 3],
        [best_outer_center[0] + 0.05, best_outer_center[1], best_outer_angle],
        [best_outer_center[0] - 0.05, best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1] + 0.05, best_outer_angle],
        [best_outer_center[0], best_outer_center[1] - 0.05, best_outer_angle],
        [best_outer_center[0] + 0.05, best_outer_center[1] + 0.05, best_outer_angle],
        [best_outer_center[0] - 0.05, best_outer_center[1] - 0.05, best_outer_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='L-BFGS-B',
                          options={'maxiter': 5000, 'gtol': 1e-12, 'maxls': 50})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
