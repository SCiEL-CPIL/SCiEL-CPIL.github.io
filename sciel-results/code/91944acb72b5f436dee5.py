import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    verts_x = cx + side * np.cos(angles)
    verts_y = cy + side * np.sin(angles)
    return np.column_stack((verts_x, verts_y))

def point_in_hexagon(px, py, hx, hy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (hx, hy)."""
    dx = px - hx
    dy = py - hy
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    inradius = side * math.sqrt(3) / 2
    for k in range(6):
        normal_angle = math.radians(30 + k * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        if lx * nx + ly * ny > inradius + 1e-9:
            return False
    return True

def hexes_overlap_fast(v1, v2):
    """Check if two hexagons overlap using vertex-in-polygon and edge intersection."""
    # Check if any vertex of v1 is inside v2
    for px, py in v1:
        if point_in_hexagon(px, py, *v2[0], v2[1], 1.0, v2[2]):
            return True
    # Check if any vertex of v2 is inside v1
    for px, py in v2:
        if point_in_hexagon(px, py, *v1[0], v1[1], 1.0, v1[2]):
            return True
    # Check edge intersections
    for i in range(6):
        p1 = v1[i]
        q1 = v1[(i + 1) % 6]
        for j in range(6):
            p2 = v2[j]
            q2 = v2[(j + 1) % 6]
            if segments_intersect(p1, q1, p2, q2):
                return True
    return False

def segments_intersect(p1, q1, p2, q2):
    """Check if segment p1-q1 intersects with segment p2-q2."""
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else 2
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        return True
    
    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    
    return False

def objective(params):
    """Objective function: outer side length + penalties."""
    outer_cx, outer_cy, outer_side, outer_angle = params[0], params[1], params[2], params[3]
    
    if outer_side < 2.0:
        return 1e10
    
    penalty = 0.0
    
    # Check containment of all inner hexagons
    for i in range(N):
        ix = params[4 + i * 3]
        iy = params[5 + i * 3]
        iang = params[6 + i * 3]
        verts = hex_vertices(ix, iy, iang, 1.0)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                # Estimate violation
                dx = vx - outer_cx
                dy = vy - outer_cy
                angle_rad = math.radians(outer_angle)
                cos_a = math.cos(-angle_rad)
                sin_a = math.sin(-angle_rad)
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                max_proj = 0.0
                for k in range(6):
                    normal_angle = math.radians(30 + k * 60)
                    nx = math.cos(normal_angle)
                    ny = math.sin(normal_angle)
                    proj = lx * nx + ly * ny
                    if proj > max_proj:
                        max_proj = proj
                req_side = 2.0 * max_proj / math.sqrt(3)
                if req_side > outer_side:
                    penalty += (req_side - outer_side) * 5.0
                else:
                    penalty += 0.5
    
    if penalty > 20.0:
        return outer_side + penalty
    
    # Check overlaps
    inner_info = []
    for i in range(N):
        ix = params[4 + i * 3]
        iy = params[5 + i * 3]
        iang = params[6 + i * 3]
        verts = hex_vertices(ix, iy, iang, 1.0)
        inner_info.append(((ix, iy), iang, verts))
    
    overlap_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            cx1, cy1 = inner_info[i][0]
            cx2, cy2 = inner_info[j][0]
            dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
            if dist_sq > 5.0:
                continue
            if hexes_overlap_fast(inner_info[i][2], inner_info[j][2]):
                overlap_count += 1
                if overlap_count > 3:
                    return outer_side + 50.0
    
    if overlap_count > 0:
        return outer_side + 5.0 * overlap_count
    
    return outer_side

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # Bounds for optimization
    bounds = [
        (-8.0, 8.0),   # outer_cx
        (-8.0, 8.0),   # outer_cy
        (3.0, 15.0),   # outer_side
        (0.0, 60.0)    # outer_angle
    ]
    for i in range(N):
        bounds.append((-8.0, 8.0))   # inner x
        bounds.append((-8.0, 8.0))   # inner y
        bounds.append((0.0, 60.0))   # inner angle
    
    # Generate good initial configurations
    def make_honeycomb(scale=1.0):
        """Create a honeycomb lattice configuration."""
        params = [0.0, 0.0, 7.0, 0.0]
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * sqrt3 * scale + (sqrt3 * scale / 2 if row % 2 == 1 else 0)
                y = row * 1.5 * scale
                x -= 2.6 * scale
                y -= 1.5 * scale
                positions.append((x, y))
        for x, y in positions:
            params.extend([x, y, 0.0])
        return np.array(params)
    
    def make_ring(radius):
        """Create a ring configuration."""
        params = [0.0, 0.0, 7.0, 0.0]
        # Center
        params.extend([0.0, 0.0, 0.0])
        # Inner ring of 6
        for k in range(6):
            a = k * math.pi / 3
            x = radius * math.cos(a)
            y = radius * math.sin(a)
            params.extend([x, y, 0.0])
        # Outer 5
        for k in range(5):
            a = math.pi/6 + k * 2*math.pi/5
            r2 = radius * 1.8
            x = r2 * math.cos(a)
            y = r2 * math.sin(a)
            params.extend([x, y, 0.0])
        return np.array(params)
    
    def make_cluster():
        """Create a compact cluster."""
        params = [0.0, 0.0, 7.0, 0.0]
        positions = [
            (0, 0),
            (sqrt3, 0), (-sqrt3, 0),
            (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
            (sqrt3/2, -1.5), (-sqrt3/2, -1.5),
            (2*sqrt3, 0),
            (sqrt3, 3.0), (-sqrt3, 3.0),
            (sqrt3, -3.0), (-sqrt3, -3.0),
        ]
        for x, y in positions:
            params.extend([x, y, 0.0])
        return np.array(params)
    
    # Try differential evolution first for global exploration
    best_result = None
    best_fun = float('inf')
    
    if time.time() < deadline - 60:
        try:
            result_de = differential_evolution(
                objective, bounds,
                seed=42, maxiter=50, tol=1e-4,
                popsize=15, mutation=(0.3, 0.8), recombination=0.7,
                polish=True, init='sobol'
            )
            if result_de.fun < best_fun:
                best_fun = result_de.fun
                best_result = result_de.x
        except Exception:
            pass
    
    # Try multiple local optimizations from different starting points
    initial_configs = [
        make_honeycomb(0.9),
        make_honeycomb(1.0),
        make_honeycomb(1.1),
        make_ring(1.8),
        make_ring(2.0),
        make_ring(2.2),
        make_cluster(),
    ]
    
    # Add some random configurations
    for _ in range(3):
        params = np.array([0.0, 0.0, 7.0, 0.0])
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            ang = np.random.uniform(0, 60)
            params = np.append(params, [x, y, ang])
        initial_configs.append(params)
    
    for init_idx, init_params in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        try:
            # Nelder-Mead
            result_nm = minimize(
                objective, init_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            if result_nm.fun < best_fun:
                best_fun = result_nm.fun
                best_result = result_nm.x
        except Exception:
            pass
        
        if time.time() > deadline - 15:
            break
        
        try:
            # L-BFGS-B
            result_lbfgs = minimize(
                objective, init_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-5, 'gtol': 1e-4}
            )
            if result_lbfgs.fun < best_fun:
                best_fun = result_lbfgs.fun
                best_result = result_lbfgs.x
        except Exception:
            pass
    
    # Final refinement if we have a good result
    if best_result is not None and best_fun < float('inf') and time.time() < deadline - 10:
        try:
            result_final = minimize(
                objective, best_result,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if result_final.fun < best_fun:
                best_fun = result_final.fun
                best_result = result_final.x
        except Exception:
            pass
    
    # Fallback if no valid result found
    if best_result is None or best_fun >= float('inf'):
        inner = []
        for row in range(3):
            for col in range(4):
                x = col * sqrt3 + (sqrt3 / 2 if row % 2 == 1 else 0)
                y = row * 1.5
                x -= 2.6
                y -= 1.5
                inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 8.0, 0.0
    
    outer_cx, outer_cy, outer_side, outer_angle = best_result[0], best_result[1], best_result[2], best_result[3]
    
    # Verify validity and fix if needed
    def check_validity(params):
        outer_cx, outer_cy, outer_side, outer_angle = params[0], params[1], params[2], params[3]
        if outer_side < 0.5:
            return False
        
        inner_verts_list = []
        for i in range(N):
            ix = params[4 + i * 3]
            iy = params[5 + i * 3]
            iang = params[6 + i * 3]
            verts = hex_vertices(ix, iy, iang, 1.0)
            inner_verts_list.append(verts)
            
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    return False
        
        for i in range(N):
            for j in range(i + 1, N):
                cx1 = params[4 + i * 3]
                cy1 = params[5 + i * 3]
                cx2 = params[4 + j * 3]
                cy2 = params[5 + j * 3]
                dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
                if dist_sq > 5.0:
                    continue
                if hexes_overlap_fast(inner_verts_list[i], inner_verts_list[j]):
                    return False
        
        return True
    
    if not check_validity(best_result):
        # Try to fix by expanding outer side
        for trial in range(20):
            new_side = outer_side + 0.1 * (trial + 1)
            test_params = list(best_result)
            test_params[2] = new_side
            if check_validity(test_params):
                best_result = test_params
                break
    
    outer_cx, outer_cy, outer_side, outer_angle = best_result[0], best_result[1], best_result[2], best_result[3]
    
    inner = []
    for i in range(N):
        ix = best_result[4 + i * 3]
        iy = best_result[5 + i * 3]
        iang = best_result[6 + i * 3]
        inner.append([ix, iy, iang])
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle

# EVOLVE-BLOCK-END
