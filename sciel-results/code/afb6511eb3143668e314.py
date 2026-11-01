import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

def hexagon_vertices(cx, cy, angle_deg, scale=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append([cx + scale * np.cos(a), cy + scale * np.sin(a)])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, angle_deg, scale=1.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    ca, sa = np.cos(-angle), np.sin(-angle)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Check against each edge
    for i in range(6):
        a1 = i * np.pi / 3
        a2 = (i + 1) * np.pi / 3
        # Edge from vertex i to vertex i+1
        v1 = np.array([scale * np.cos(a1), scale * np.sin(a1)])
        v2 = np.array([scale * np.cos(a2), scale * np.sin(a2)])
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        # Ensure normal points inward
        if np.dot(normal, -v1) < 0:
            normal = -normal
        dist = np.dot(np.array([lx, ly]) - v1, normal)
        if dist < -1e-9:
            return False
    return True

def hexagon_distance(v1, v2):
    """
    Minimum distance between two hexagons.
    v1, v2: (cx, cy, angle_deg) tuples
    Returns minimum distance between the two hexagon boundaries (0 if overlapping).
    """
    cx1, cy1, a1 = v1
    cx2, cy2, a2 = v2
    dx = cx2 - cx1
    dy = cy2 - cy1
    d = np.sqrt(dx*dx + dy*dy)
    
    # Quick check: if centers are very close, definitely overlapping
    if d < 1e-6:
        return -1.0  # overlapping
    
    # The minimum distance between two unit hexagons depends on relative rotation
    # Use a conservative bound: minimum center distance for non-overlap is sqrt(3)
    # But for actual distance computation, we need to check edges
    
    # Get vertices of both hexagons
    verts1 = hexagon_vertices(cx1, cy1, a1)
    verts2 = hexagon_vertices(cx2, cy2, a2)
    
    # Check if any vertex of one is inside the other
    for i in range(6):
        if point_in_hexagon(verts1[i][0], verts1[i][1], cx2, cy2, a2):
            return -1.0  # overlapping
        if point_in_hexagon(verts2[i][0], verts2[i][1], cx1, cy1, a1):
            return -1.0  # overlapping
    
    # Check edge-edge distances
    min_dist = float('inf')
    for i in range(6):
        e1 = [verts1[i], verts1[(i+1)%6]]
        for j in range(6):
            e2 = [verts2[j], verts2[(j+1)%6]]
            # Distance between two line segments
            d_seg = seg_seg_dist(e1[0], e1[1], e2[0], e2[1])
            min_dist = min(min_dist, d_seg)
    
    return min_dist

def seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-10 and e < 1e-10:
        return np.linalg.norm(r)
    elif a < 1e-10:
        t2 = max(0, min(1, f / e))
        closest = p3 + t2 * d2
        return np.linalg.norm(p1 - closest)
    elif e < 1e-10:
        t1 = max(0, min(1, -np.dot(d1, r) / a))
        closest = p1 + t1 * d1
        return np.linalg.norm(p3 - closest)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom < 1e-10:
        t1 = max(0, min(1, (b - f) / a))
        closest1 = p1 + t1 * d1
        t2 = max(0, min(1, (b * t1 + c - f) / e)) if e > 1e-10 else 0
        closest2 = p3 + t2 * d2
        return min(np.linalg.norm(closest1 - p3), np.linalg.norm(p1 - closest2), np.linalg.norm(closest1 - closest2))
    else:
        t1 = max(0, min(1, (b * f - c * e) / denom))
        t2 = max(0, min(1, (a * f - b * c) / denom))
        closest1 = p1 + t1 * d1
        closest2 = p3 + t2 * d2
        return np.linalg.norm(closest1 - closest2)

def max_extent(hexagons, outer_cx, outer_cy, outer_angle, outer_side):
    """
    Check if all hexagons fit inside the outer hexagon.
    Returns the maximum violation (0 if all fit, positive if some stick out).
    """
    max_viol = 0.0
    for (cx, cy, angle) in hexagons:
        verts = hexagon_vertices(cx, cy, angle)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_angle, outer_side):
                # Compute how far outside
                # Find the minimum distance to the outer hexagon boundary
                outer_verts = hexagon_vertices(outer_cx, outer_cy, outer_angle, outer_side)
                min_d = float('inf')
                for i in range(6):
                    d = seg_seg_dist([vx, vy], [vx, vy], outer_verts[i], outer_verts[(i+1)%6])
                    min_d = min(min_d, d)
                max_viol = max(max_viol, min_d)
    return max_viol

def compute_objective(params, N):
    """
    params: array of length 3*N + 3
    First 3*N: (cx, cy, angle) for each inner hexagon
    Last 3: outer_cx, outer_cy, outer_side (outer angle fixed at 0)
    Returns: outer_side + penalty for overlaps and containment violations
    """
    inner = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        angle = params[3*i+2]
        inner.append((cx, cy, angle))
    
    outer_cx = params[3*N]
    outer_cy = params[3*N+1]
    outer_side = params[3*N+2]
    
    penalty = 0.0
    
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i+1, N):
            d = hexagon_distance(inner[i], inner[j])
            if d < 0:  # overlapping
                penalty += 10.0 * (-d)
            elif d < 1e-6:
                penalty += 1.0
    
    # Check containment
    viol = max_extent(inner, outer_cx, outer_cy, 0.0, outer_side)
    if viol > 0:
        penalty += 100.0 * viol
    
    return outer_side + penalty

def initial_config_hexagonal(N):
    """Create a hexagonal lattice initial configuration."""
    inner = []
    # Hexagonal lattice spacing: sqrt(3) for edge-to-edge contact
    sp = np.sqrt(3) * 1.001  # slight gap
    
    # Generate hexagonal positions
    positions = []
    # Center
    positions.append((0, 0))
    # Ring 1: 6 positions
    for k in range(6):
        a = k * np.pi / 3
        positions.append((sp * np.cos(a), sp * np.sin(a)))
    # Ring 2: 12 positions (we need 4 more for total 11)
    for k in range(12):
        a = k * np.pi / 6
        r = sp * np.sqrt(3)
        positions.append((r * np.cos(a), r * np.sin(a)))
    
    # Take first N positions
    for i in range(min(N, len(positions))):
        inner.append([positions[i][0], positions[i][1], 0.0])
    
    while len(inner) < N:
        inner.append([0.0, 0.0, 0.0])
    
    return inner

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score = float('inf')
    
    # Initial hexagonal configuration
    inner_init = initial_config_hexagonal(N)
    
    # Estimate initial outer size
    max_r = max(np.sqrt(p[0]**2 + p[1]**2) for p in inner_init)
    initial_outer = max_r + 2.0  # rough estimate
    
    # Build initial parameter vector
    params0 = []
    for (cx, cy, angle) in inner_init:
        params0.extend([cx, cy, angle])
    params0.extend([0.0, 0.0, initial_outer])
    params0 = np.array(params0)
    
    # Try multiple initial configurations and optimize
    configs_to_try = []
    
    # Config 1: Standard hexagonal lattice
    configs_to_try.append(inner_init.copy())
    
    # Config 2: Linear arrangement
    inner_lin = []
    sp = np.sqrt(3) * 1.001
    for i in range(N):
        inner_lin.append([(i - N//2) * sp, 0.0, 0.0])
    configs_to_try.append(inner_lin)
    
    # Config 3: Two rows
    inner_two = []
    sp = np.sqrt(3) * 1.001
    for i in range(N):
        row = i // 6
        col = i % 6
        x = (col - 2.5) * sp
        y = row * sp * np.sqrt(3) / 2
        inner_two.append([x, y, 0.0])
    configs_to_try.append(inner_two)
    
    # Config 4: Compact cluster
    inner_cluster = []
    sp = np.sqrt(3) * 1.001
    # 3x4 grid with hexagonal offset
    for i in range(4):
        for j in range(3):
            if len(inner_cluster) >= N:
                break
            x = (j - 1) * sp + (i % 2) * sp/2
            y = (i - 1.5) * sp * np.sqrt(3)/2
            inner_cluster.append([x, y, 0.0])
    while len(inner_cluster) < N:
        inner_cluster.append([0.0, 0.0, 0.0])
    configs_to_try.append(inner_cluster)
    
    # Config 5: Random
    np.random.seed(42)
    inner_rand = []
    for i in range(N):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        angle = np.random.uniform(0, 60)
        inner_rand.append([x, y, angle])
    configs_to_try.append(inner_rand)
    
    # Config 6: Hexagonal with some rotation
    inner_hex_rot = []
    sp = np.sqrt(3) * 1.001
    positions = [(0, 0)]
    for k in range(6):
        a = k * np.pi / 3
        positions.append((sp * np.cos(a), sp * np.sin(a)))
    for k in range(4):
        a = k * np.pi / 2
        r = sp * np.sqrt(3)
        positions.append((r * np.cos(a), r * np.sin(a)))
    for i in range(N):
        inner_hex_rot.append([positions[i][0], positions[i][1], 0.0])
    configs_to_try.append(inner_hex_rot)
    
    for config_idx, inner_cfg in enumerate(configs_to_try):
        if time.time() > deadline:
            break
        
        # Build parameter vector
        max_r = max(np.sqrt(p[0]**2 + p[1]**2) for p in inner_cfg)
        outer_est = max_r + 2.0
        
        params = []
        for (cx, cy, angle) in inner_cfg:
            params.extend([cx, cy, angle])
        params.extend([0.0, 0.0, outer_est])
        params = np.array(params)
        
        # Use Nelder-Mead optimization
        try:
            result = minimize(
                compute_objective,
                params,
                args=(N,),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        # Extract result
        best_params = result.x
        inner_res = []
        for i in range(N):
            inner_res.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
        outer_cx = best_params[3*N]
        outer_cy = best_params[3*N+1]
        outer_side = best_params[3*N+2]
        
        # Verify validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = hexagon_distance(inner_res[i], inner_res[j])
                if d < -1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Check containment
            viol = max_extent(inner_res, outer_cx, outer_cy, 0.0, outer_side)
            if viol < 1e-4:
                score = outer_side
                if score < best_score:
                    best_score = score
                    best_result = (inner_res, (outer_cx, outer_cy), outer_side, 0.0)
        
        # Polish with tighter tolerance if time permits
        if time.time() < deadline - 20 and best_result is not None:
            try:
                result2 = minimize(
                    compute_objective,
                    best_params,
                    args=(N,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                best_params2 = result2.x
                inner_res2 = []
                for i in range(N):
                    inner_res2.append([best_params2[3*i], best_params2[3*i+1], best_params2[3*i+2]])
                outer_cx2 = best_params2[3*N]
                outer_cy2 = best_params2[3*N+1]
                outer_side2 = best_params2[3*N+2]
                
                # Verify
                valid2 = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = hexagon_distance(inner_res2[i], inner_res2[j])
                        if d < -1e-6:
                            valid2 = False
                            break
                    if not valid2:
                        break
                
                if valid2:
                    viol2 = max_extent(inner_res2, outer_cx2, outer_cy2, 0.0, outer_side2)
                    if viol2 < 1e-4:
                        score2 = outer_side2
                        if score2 < best_score:
                            best_score = score2
                            best_result = (inner_res2, (outer_cx2, outer_cy2), outer_side2, 0.0)
            except Exception:
                pass
    
    # If no valid result found, return a safe default
    if best_result is None:
        # Return the seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-START
# (construct function is defined above)
# EVOLVE-BLOCK-END
