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

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """
    Compute a smooth penalty for overlaps based on center distances.
    This is a proxy for actual overlap, valid for aligned hexagons.
    """
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    epsilon = 1e-6
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

    # 3. Optimize Outer Hexagon using scipy with multiple starts
    # Variables: [cx, cy, angle_deg]
    
    def objective_outer(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Initial guess: centroid and 0
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    init_angles = [0, 10, 20, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    for init_angle in init_angles:
        if time.time() > deadline - 20:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead', 
                          options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 1500})
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
        # Fallback to grid search if scipy fails or times out
        best_side = float('inf')
        for outer_angle_deg in range(0, 360, 5):
            current_cx = cx_avg
            current_cy = cy_avg
            step = 0.5
            for _ in range(3):
                improved = False
                for dx in np.linspace(-step, step, 5):
                    for dy in np.linspace(-step, step, 5):
                        oc = (current_cx + dx, current_cy + dy)
                        side = compute_required_side(initial_inner, oc, outer_angle_deg)
                        if side < best_side - 1e-6:
                            best_side = side
                            best_outer_center = oc
                            best_outer_angle = outer_angle_deg
                            current_cx = oc[0]
                            current_cy = oc[1]
                            improved = True
                if not improved:
                    break
                step *= 0.5
            if time.time() > deadline - 5:
                break

    # 4. Joint Optimization of Inner Positions and Outer Container
    # We treat the outer center and angle as part of the optimization variables
    # to allow the container to adapt to the inner configuration.
    
    def objective_joint(params):
        # Unpack params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, angle]
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        oc = (params[N*3], params[N*3+1])
        oa = params[N*3+2]
        
        side = compute_required_side(inner_list, oc, oa)
        
        # Penalty for overlaps
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 500.0 * penalty
    
    # Initial params: initial_inner + best_outer
    current_params = np.array([v for inner in initial_inner for v in inner] + 
                              [best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Simulated Annealing for joint optimization
    T = 0.5
    T_min = 0.005
    cooling = 0.9995
    max_iters = 8000
    
    for i in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Generate neighbor
        neighbor = current_params.copy()
        # Randomly pick a dimension to perturb
        # With higher probability, perturb inner positions
        if np.random.random() < 0.8:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            offset = idx * 3 + dim
            # Larger steps for position, smaller for angle
            if dim < 2:
                delta = np.random.normal(0, 0.5)
            else:
                delta = np.random.normal(0, 5.0)
        else:
            # Perturb outer container
            dim = np.random.randint(0, 3)
            offset = N * 3 + dim
            if dim < 2:
                delta = np.random.normal(0, 0.2)
            else:
                delta = np.random.normal(0, 2.0)
        
        neighbor[offset] += delta
        
        neighbor_val = objective_joint(neighbor)
        
        # Accept if better or with probability exp(-delta/T)
        if neighbor_val < current_val or (neighbor_val - current_val) < 100 and np.random.random() < math.exp(-(neighbor_val - current_val) / T):
            current_params = neighbor
            current_val = neighbor_val
            
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
        
        T *= cooling
        if T < T_min:
            # Restart with a slightly higher temperature to escape local minima
            if i < max_iters * 0.8:
                T = 0.2
            else:
                break
    
    # Reconstruct best inner and outer
    best_inner_list = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_list.append([x, y, a])
    
    best_outer_center = (best_params[N*3], best_params[N*3+1])
    best_outer_angle = best_params[N*3+2]
    
    # Verify with SAT
    if check_all_disjoint(best_inner_list, eps=1e-5):
        best_inner = best_inner_list
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        # If SAT fails, try to repair by pushing overlapping hexagons apart slightly
        # Or fallback to initial if repair is too complex
        # For now, fallback to the best valid configuration found during SA if any
        # We didn't track valid intermediate states, so we fallback to initial lattice
        # which is guaranteed valid (with epsilon scale)
        best_inner = initial_inner
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
        
        # Re-optimize outer for the fallback inner
        def objective_outer_fallback(params):
            cx, cy, angle_deg = params
            side = compute_required_side(initial_inner, (cx, cy), angle_deg)
            return side
            
        try:
            res = minimize(objective_outer_fallback, [best_outer_center[0], best_outer_center[1], best_outer_angle], 
                          method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000})
            if res.fun < best_side:
                best_outer_center = (res.x[0], res.x[1])
                best_outer_angle = res.x[2]
                best_side = res.fun
        except Exception:
            pass

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
