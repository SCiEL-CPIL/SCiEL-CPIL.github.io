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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp soft-max.
    This differentiable approximation helps gradient-based optimizers.
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
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    proj_arr = np.array(projections)
    max_proj = float(np.log(np.sum(np.exp(beta * proj_arr))) / beta)
    
    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
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

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Compute a smooth penalty for overlaps based on center distances."""
    n = len(inner)
    if n < 2:
        return 0.0
    
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    
    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff * diff))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Compact 3-4-3 hexagonal cluster: center, 6 neighbors, 4 more forming tight ring
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
    
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (cx_avg, cy_avg)
    best_outer_angle = 0.0
    
    # Phase 1: Optimize outer container with fixed inner layout
    # Two-phase: smooth surrogate for basin finding, then exact for precision
    
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
    
    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        return compute_required_side(initial_inner, (cx, cy), angle_deg)
    
    best_result = None
    best_val = float('inf')
    
    init_angles = list(range(0, 360, 15))
    
    for init_angle in init_angles:
        if time.time() > deadline - 40:
            break
        
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 3000})
            
            res2 = minimize(objective_outer_exact, res.x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
            
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x
        except Exception:
            pass
    
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    
    # Phase 2: Joint optimization with simulated annealing
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
        
        side = compute_required_side(inner_list, (ocx, ocy), oangle)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_params = np.array(initial_params)
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    T = 1.0
    T_min = 0.005
    cooling = 0.9998
    
    max_iter = 15000
    for i in range(max_iter):
        if time.time() > deadline - 15:
            break
        
        neighbor = current_params.copy()
        
        if np.random.random() < 0.8:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            delta = np.random.normal(0, 0.2)
            neighbor[idx*3 + dim] += delta
        else:
            dim = np.random.randint(N*3, N*3+3)
            if dim == N*3+2:
                delta = np.random.normal(0, 3.0)
            else:
                delta = np.random.normal(0, 0.15)
            neighbor[dim] += delta
        
        neighbor_val = objective_joint(neighbor)
        
        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / T):
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
        best_inner = initial_inner
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # Phase 3: Final high-precision polish on outer container
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 2],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 2],
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
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 8000})
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
