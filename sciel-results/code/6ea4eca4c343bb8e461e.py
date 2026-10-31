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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    This provides better gradient information for optimization.
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
    
    # Collect all projections
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
    
    # Log-sum-exp for smooth max
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    # Smooth max via log-sum-exp: (1/beta) * log(sum(exp(beta * p_i)))
    # This approximates max(p_i) as beta -> infinity
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

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

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice - use tighter, more symmetric arrangement
    # Center hexagon + 6 around it + 4 more in a compact cluster
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian with tight packing
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
    # Use smooth surrogate for better optimization landscape
    
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        side = compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
        return side
    
    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Initial guess: centroid and 0
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    for init_angle in init_angles:
        if time.time() > deadline - 20:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            # First pass: smooth optimization for basin finding
            res_smooth = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000})
            
            # Second pass: exact optimization from smooth result
            if res_smooth.fun < best_val or best_result is None:
                res = minimize(objective_outer_exact, res_smooth.x, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
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

    # 4. Optimize inner positions with SAT-based validity checking
    # Use a penalty approach but verify with SAT at the end
    
    def objective_inner(params):
        # Unpack params to inner list
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        side = compute_required_side(inner_list, best_outer_center, best_outer_angle)
        
        # Conservative penalty: center distance < sqrt(3) - margin
        penalty = 0.0
        min_dist = math.sqrt(3) - 0.01  # Safe margin
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return side + 1000.0 * penalty
    
    # Simulated Annealing for inner positions with adaptive step size
    current_params = np.array([v for inner in initial_inner for v in inner])
    current_val = objective_inner(current_params)
    best_params = current_params.copy()
    best_val_inner = current_val
    
    T = 1.0
    T_min = 0.005
    cooling = 0.9985
    step_size = 0.4
    no_improve_count = 0
    max_no_improve = 200
    
    for i in range(8000):
        if time.time() > deadline - 10:
            break
        
        # Generate neighbor with adaptive step size
        neighbor = current_params.copy()
        idx = np.random.randint(0, N)
        dim = np.random.randint(0, 3)
        delta = np.random.normal(0, step_size)
        neighbor[idx*3 + dim] += delta
        
        neighbor_val = objective_inner(neighbor)
        
        # Accept if better or with probability exp(-delta/T)
        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, 1e-10)):
            current_params = neighbor
            current_val = neighbor_val
            
            if current_val < best_val_inner:
                best_params = current_params.copy()
                best_val_inner = current_val
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Reduce step size if no improvement for a while
            if no_improve_count > max_no_improve:
                step_size *= 0.8
                no_improve_count = 0
        else:
            no_improve_count += 1
        
        T *= cooling
        if T < T_min:
            break
    
    # Reconstruct best inner
    best_inner_list = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_list.append([x, y, a])
    
    # Verify with SAT
    if check_all_disjoint(best_inner_list, eps=1e-5):
        best_inner = best_inner_list
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        # Fallback to initial if SAT fails
        best_inner = initial_inner
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # 5. Final high-precision polish pass
    if time.time() < deadline - 5:
        # Polish outer parameters with tighter tolerance
        try:
            def polish_objective(params):
                cx, cy, angle_deg = params
                return compute_required_side(best_inner, (cx, cy), angle_deg)
            
            polish_result = minimize(polish_objective, [best_outer_center[0], best_outer_center[1], best_outer_angle],
                                    method='Nelder-Mead',
                                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
            
            if polish_result.fun < best_side:
                best_outer_center = (polish_result.x[0], polish_result.x[1])
                best_outer_angle = polish_result.x[2]
                best_side = polish_result.fun
        except Exception:
            pass
        
        # Polish inner positions with scipy from current best
        if time.time() < deadline - 3:
            try:
                def polish_inner_objective(params):
                    inner_list = []
                    for i in range(N):
                        x = params[i*3]
                        y = params[i*3+1]
                        a = params[i*3+2]
                        inner_list.append([x, y, a])
                    
                    side = compute_required_side(inner_list, best_outer_center, best_outer_angle)
                    
                    penalty = 0.0
                    min_dist = math.sqrt(3) - 0.005
                    for i in range(N):
                        for j in range(i+1, N):
                            xi, yi, _ = inner_list[i]
                            xj, yj, _ = inner_list[j]
                            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                            if dist < min_dist:
                                penalty += (min_dist - dist) ** 2
                    
                    return side + 10000.0 * penalty
                
                inner_polish = minimize(polish_inner_objective, best_params, method='Nelder-Mead',
                                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 8000})
                
                if inner_polish.fun < best_side:
                    polished_inner = []
                    for i in range(N):
                        x = inner_polish.x[i*3]
                        y = inner_polish.x[i*3+1]
                        a = inner_polish.x[i*3+2]
                        polished_inner.append([x, y, a])
                    
                    if check_all_disjoint(polished_inner, eps=1e-6):
                        best_inner = polished_inner
                        best_side = compute_required_side(polished_inner, best_outer_center, best_outer_angle)
            except Exception:
                pass
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
