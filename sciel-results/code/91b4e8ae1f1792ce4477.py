import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
TIME_LIMIT = 140.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return the 6 vertices of a regular hexagon with given center, rotation, and side length."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _hex_edge_normals(angle_deg):
    """Return the 6 outward edge normals of a unit hexagon with given rotation."""
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + math.radians(60 * k + 90)
        normals.append((math.cos(theta), math.sin(theta)))
    return normals

def _sat_overlap_fast(verts_a, verts_b, normals_a, normals_b):
    """
    Fast SAT check: returns True if two convex polygons overlap.
    Uses early termination.
    """
    # Check all axes from polygon A
    for nx, ny in normals_a:
        min_a, max_a = float('inf'), float('-inf')
        for vx, vy in verts_a:
            proj = vx * nx + vy * ny
            if proj < min_a: min_a = proj
            if proj > max_a: max_a = proj
        
        min_b, max_b = float('inf'), float('-inf')
        for vx, vy in verts_b:
            proj = vx * nx + vy * ny
            if proj < min_b: min_b = proj
            if proj > max_b: max_b = proj
        
        if max_a <= min_b + 1e-9 or max_b <= min_a + 1e-9:
            return False  # Separated along this axis
    
    # Check all axes from polygon B
    for nx, ny in normals_b:
        min_a, max_a = float('inf'), float('-inf')
        for vx, vy in verts_a:
            proj = vx * nx + vy * ny
            if proj < min_a: min_a = proj
            if proj > max_a: max_a = proj
        
        min_b, max_b = float('inf'), float('-inf')
        for vx, vy in verts_b:
            proj = vx * nx + vy * ny
            if proj < min_b: min_b = proj
            if proj > max_b: max_b = proj
        
        if max_a <= min_b + 1e-9 or max_b <= min_a + 1e-9:
            return False  # Separated along this axis
    
    return True  # No separating axis found, so they overlap

def _check_disjoint(cx_arr, cy_arr, angle_arr):
    """Check if all hexagons are pairwise disjoint using SAT."""
    n = len(cx_arr)
    verts_list = []
    normals_list = []
    
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
        normals_list.append(_hex_edge_normals(angle_arr[i]))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Quick distance check
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            
            # Maximum distance between centers for non-overlapping unit hexagons
            # is 2 * circumradius = 2.0, but for safety use a slightly larger bound
            if dist_sq > 4.5:  # 2.12^2, safe upper bound
                continue
            
            if _sat_overlap_fast(verts_list[i], verts_list[j], 
                                  normals_list[i], normals_list[j]):
                return False
    
    return True

def _outer_hex_side_fast(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """
    Compute the minimum side length of a regular hexagon (with given center and rotation)
    that contains all inner hexagons.
    """
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    
    return max_need

def _make_hex_grid():
    """Create a hexagonal grid arrangement of 12 hexagons."""
    pts = []
    # Center
    pts.append([0.0, 0.0, 0.0])
    # First ring (6 hexagons)
    r1 = 2 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        pts.append([x, y, 0.0])
    # Second ring (5 more to reach 12)
    r2 = r1 * 1.732
    angles = [30, 90, 150, 210, 270]
    for deg in angles:
        rad = math.radians(deg)
        x = r2 * math.cos(rad)
        y = r2 * math.sin(rad)
        pts.append([x, y, 0.0])
    return pts[:12]

def _make_ring():
    """Create a ring arrangement of 12 hexagons around a center."""
    pts = [[0.0, 0.0, 0.0]]
    r = 2 * math.sqrt(3)
    for i in range(11):
        angle = math.radians(360 * i / 11 + 15)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        pts.append([x, y, 0.0])
    return pts

def _make_cluster():
    """Create a compact cluster arrangement."""
    pts = []
    # 3x4 rectangular grid with hexagonal offset
    row_height = math.sqrt(3)
    col_width = 2.0
    for row in range(3):
        y = row * row_height - row_height
        offset = (col_width / 2) if row % 2 == 1 else 0.0
        for col in range(4):
            x = col * col_width - col_width + offset
            pts.append([x, y, 0.0])
    return pts[:12]

def _make_strip():
    """Create a strip arrangement."""
    pts = []
    # Two rows of 6
    row_height = math.sqrt(3)
    col_width = 2.0
    for row in range(2):
        y = row * row_height - row_height / 2
        offset = (col_width / 2) if row % 2 == 1 else 0.0
        for col in range(6):
            x = col * col_width - 2.5 * col_width + offset
            pts.append([x, y, 0.0])
    return pts

def _physics_relax(cx_arr, cy_arr, angle_arr, iterations=200):
    """
    Simple physics-based relaxation to resolve overlaps.
    Returns relaxed positions.
    """
    cx = cx_arr.copy()
    cy = cy_arr.copy()
    ang = angle_arr.copy()
    
    for _ in range(iterations):
        # Compute repulsion forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = math.hypot(dx, dy)
                
                if dist < 1e-6:
                    # Random perturbation
                    dx = random.uniform(-0.01, 0.01)
                    dy = random.uniform(-0.01, 0.01)
                    dist = math.hypot(dx, dy)
                
                # Minimum distance for non-overlap
                min_dist = 2.0  # Approximate for unit hexagons
                
                if dist < min_dist:
                    # Repulsive force
                    force = (min_dist - dist) * 10.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Apply forces
        for i in range(N):
            cx[i] += fx[i] * 0.1
            cy[i] += fy[i] * 0.1
    
    return cx, cy, ang

def _objective(params, cx_arr, cy_arr, angle_arr, center, angle_deg, penalty_weight=100.0):
    """
    Objective function: outer hex side length + penalty for overlaps.
    params: [x1, y1, a1, ..., xN, yN, aN]
    """
    xs = params[0:N]
    ys = params[N:2*N]
    angles = params[2*N:3*N]
    
    # Compute outer hex side
    side = _outer_hex_side_fast(xs, ys, angles, center, angle_deg)
    
    # Compute overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.hypot(dx, dy)
            
            # Approximate minimum distance
            min_dist = 2.0
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += overlap * overlap
    
    return side + penalty_weight * penalty

def _objective_nm(params, center, angle_deg, penalty_weight=100.0):
    """
    Objective function for Nelder-Mead (no pre-computed arrays).
    params: [x1, y1, a1, ..., xN, yN, aN]
    """
    xs = params[0:N]
    ys = params[N:2*N]
    angles = params[2*N:3*N]
    
    # Compute outer hex side
    side = _outer_hex_side_fast(xs, ys, angles, center, angle_deg)
    
    # Compute overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.hypot(dx, dy)
            
            min_dist = 2.0
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += overlap * overlap
    
    return side + penalty_weight * penalty

def _polish(cx_arr, cy_arr, angle_arr, center, angle_deg, time_budget=10.0):
    """
    Polish a configuration using L-BFGS-B and Nelder-Mead.
    Returns improved configuration and side length.
    """
    t0 = time.time()
    
    # Convert to parameter vector
    params = np.zeros(3 * N)
    for i in range(N):
        params[i] = cx_arr[i]
        params[N + i] = cy_arr[i]
        params[2 * N + i] = angle_arr[i]
    
    best_side = _outer_hex_side_fast(cx_arr, cy_arr, angle_arr, center, angle_deg)
    best_params = params.copy()
    
    # L-BFGS-B optimization
    if time.time() - t0 < time_budget - 2:
        try:
            result = minimize(
                lambda p: _objective(p, cx_arr, cy_arr, angle_arr, center, angle_deg),
                params,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_side:
                best_params = result.x.copy()
                best_side = result.fun
        except Exception:
            pass
    
    # Nelder-Mead refinement
    if time.time() - t0 < time_budget - 1:
        try:
            result = minimize(
                lambda p: _objective_nm(p, center, angle_deg),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < best_side:
                best_params = result.x.copy()
                best_side = result.fun
        except Exception:
            pass
    
    # Extract best configuration
    best_cx = best_params[0:N]
    best_cy = best_params[N:2*N]
    best_ang = best_params[2*N:3*N]
    
    # Verify disjointness
    if _check_disjoint(best_cx, best_cy, best_ang):
        side = _outer_hex_side_fast(best_cx, best_cy, best_ang, center, angle_deg)
        return best_cx, best_cy, best_ang, side
    
    return cx_arr, cy_arr, angle_arr, best_side

def construct():
    t0 = time.time()
    random.seed(42)
    
    # Generate initial configurations
    initializers = [
        _make_hex_grid,
        _make_ring,
        _make_cluster,
        _make_strip,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try each initializer
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 30:
            break
        
        cfg = init_func()
        if len(cfg) != N:
            continue
        
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        # Check disjointness
        if _check_disjoint(cx_arr, cy_arr, angle_arr):
            side = _outer_hex_side_fast(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
    # If no valid configuration found, create a fallback
    if best_cfg is None:
        cfg = _make_hex_grid()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _outer_hex_side_fast(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    # Phase 1: Physics-based relaxation to resolve any near-overlaps
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    if time.time() - t0 < TIME_LIMIT - 25:
        cx_relaxed, cy_relaxed, angle_relaxed = _physics_relax(cx_best, cy_best, angle_best, iterations=100)
        
        if _check_disjoint(cx_relaxed, cy_relaxed, angle_relaxed):
            side_relaxed = _outer_hex_side_fast(cx_relaxed, cy_relaxed, angle_relaxed, (0.0, 0.0), 0.0)
            if side_relaxed < best_side:
                best_side = side_relaxed
                best_cfg = [[cx_relaxed[i], cy_relaxed[i], angle_relaxed[i]] for i in range(N)]
                best_center = (0.0, 0.0)
                best_angle = 0.0
                cx_best, cy_best, angle_best = cx_relaxed, cy_relaxed, angle_relaxed
    
    # Phase 2: Gradient-based optimization (L-BFGS-B + Nelder-Mead)
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    if time.time() - t0 < TIME_LIMIT - 15:
        cx_opt, cy_opt, angle_opt, side_opt = _polish(
            cx_best, cy_best, angle_best, best_center, best_angle, time_budget=10.0
        )
        
        if side_opt < best_final_side:
            best_final_side = side_opt
            best_final_cfg = [[cx_opt[i], cy_opt[i], angle_opt[i]] for i in range(N)]
            best_final_center = best_center
            best_final_angle = best_angle
    
    # Phase 3: Try different outer hexagon rotations and centers
    cx_best2 = np.array([p[0] for p in best_final_cfg])
    cy_best2 = np.array([p[1] for p in best_final_cfg])
    angle_best2 = np.array([p[2] for p in best_final_cfg])
    
    for outer_angle in range(0, 30, 3):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        
        # Optimize center for this outer angle
        center = (0.0, 0.0)
        
        # Simple center optimization
        for _ in range(10):
            best_cx, best_cy, best_val = center[0], center[1], float('inf')
            
            for dx in [-0.3, -0.15, 0, 0.15, 0.3]:
                for dy in [-0.3, -0.15, 0, 0.15, 0.3]:
                    ncx, ncy = center[0] + dx, center[1] + dy
                    val = _outer_hex_side_fast(cx_best2, cy_best2, angle_best2, (ncx, ncy), float(outer_angle))
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - center[0]) < 1e-4 and abs(best_cy - center[1]) < 1e-4:
                break
            center = (best_cx, best_cy)
        
        side = _outer_hex_side_fast(cx_best2, cy_best2, angle_best2, center, float(outer_angle))
        if side < best_final_side:
            best_final_side = side
            best_final_center = center
            best_final_angle = float(outer_angle)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _check_disjoint(cx_final, cy_final, angle_final):
        # Fallback: use scaled hex grid
        cfg = _make_hex_grid()
        scale = 1.3
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_final_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_final_side = _outer_hex_side_fast(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
    else:
        best_final_side = _outer_hex_side_fast(cx_final, cy_final, angle_final, best_final_center, best_final_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_final_cfg]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
