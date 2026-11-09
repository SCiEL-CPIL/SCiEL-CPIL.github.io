"""Optimized hexagon packing via differentiable overlap penalty + L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, s, ang_deg):
    a = np.deg2rad(ang_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Regular hexagon vertices at angles 0,60,120,180,240,300
    base_ang = np.arange(6) * np.pi / 3.0
    vx = np.cos(base_ang) * s
    vy = np.sin(base_ang) * s
    # Rotate
    rx = vx * ca - vy * sa
    ry = vx * sa + vy * ca
    return np.column_stack((rx + cx, ry + cy))

def _point_in_hex(px, py, cx, cy, s, ang_deg):
    a = np.deg2rad(ang_deg)
    ca, sa = np.cos(a), np.sin(a)
    dx = px - cx
    dy = py - cy
    # Rotate point into hexagon frame
    lx = dx * ca + dy * sa
    ly = -dx * sa + dy * ca
    # Regular hexagon: |x| <= s, |y| <= s*sqrt(3)/2, |x| + |y|/sqrt(3) <= s
    absx = np.abs(lx)
    absy = np.abs(ly)
    return (absx <= s + 1e-9) & (absy <= s * np.sqrt(3) / 2 + 1e-9) & (absx + absy / np.sqrt(3) <= s + 1e-9)

def _seg_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return False
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    return -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9

def _poly_overlap_area(v1, v2):
    """Compute overlap area of two convex polygons using Sutherland-Hodgman clipping."""
    def clip_polygon(subject, clip_edge_start, clip_edge_end):
        if len(subject) == 0:
            return []
        output = []
        def inside(p):
            # Check if p is to the left of directed edge clip_edge_start -> clip_edge_end
            edge = clip_edge_end - clip_edge_start
            to_p = p - clip_edge_start
            cross = edge[0] * to_p[1] - edge[1] * to_p[0]
            return cross >= -1e-9
        def intersect(p1, p2):
            d1 = p2 - p1
            d2 = clip_edge_end - clip_edge_start
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < 1e-12:
                return p1
            t = ((clip_edge_start[0] - p1[0]) * d2[1] - (clip_edge_start[1] - p1[1]) * d2[0]) / cross
            return p1 + t * d1
        n = len(subject)
        for i in range(n):
            cur = subject[i]
            nxt = subject[(i + 1) % n]
            cur_in = inside(cur)
            nxt_in = inside(nxt)
            if cur_in:
                output.append(cur)
                if not nxt_in:
                    output.append(intersect(cur, nxt))
            elif nxt_in:
                output.append(intersect(cur, nxt))
                output.append(nxt)
        return output
    
    # Clip v1 by each edge of v2
    subject = list(v1)
    for i in range(len(v2)):
        a = v2[i]
        b = v2[(i + 1) % len(v2)]
        subject = clip_polygon(subject, a, b)
        if len(subject) == 0:
            return 0.0
    
    if len(subject) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    for i in range(len(subject)):
        x1, y1 = subject[i]
        x2, y2 = subject[(i + 1) % len(subject)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def _outer_hex_vertices(cx, cy, s, ang_deg):
    a = np.deg2rad(ang_deg)
    ca, sa = np.cos(a), np.sin(a)
    base_ang = np.arange(6) * np.pi / 3.0
    vx = np.cos(base_ang) * s
    vy = np.sin(base_ang) * s
    rx = vx * ca - vy * sa
    ry = vx * sa + vy * ca
    return np.column_stack((rx + cx, ry + cy))

def _inner_in_outer(inner_verts, outer_verts):
    """Check if all inner vertices are inside outer hexagon."""
    # For each inner vertex, check if inside outer hexagon
    # Outer hexagon is convex, so check against each edge
    for iv in inner_verts:
        for i in range(len(outer_verts)):
            a = outer_verts[i]
            b = outer_verts[(i + 1) % len(outer_verts)]
            edge = b - a
            to_p = iv - a
            cross = edge[0] * to_p[1] - edge[1] * to_p[0]
            if cross < -1e-6:
                return False
    return True

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_result = None
    best_side = float('inf')
    
    # Initial guess: honeycomb-like arrangement
    # 12 hexagons: center + ring of 6 + partial outer ring
    # Actually, let's use a grid-like initial placement
    s_inner = 1.0
    # Initial positions: try a compact arrangement
    # Hex grid spacing: for side s, center-to-center distance for touching hexes is s*sqrt(3)
    hex_dist = s_inner * np.sqrt(3)
    
    # Create initial positions in a hexagonal pattern
    # 12 hexes: 1 center, 6 in first ring, 5 in second ring (partial)
    init_positions = []
    init_angles = []
    # Center
    init_positions.append([0.0, 0.0])
    init_angles.append(0.0)
    # First ring: 6 hexes at distance hex_dist
    for k in range(6):
        ang = k * np.pi / 3.0
        init_positions.append([hex_dist * np.cos(ang), hex_dist * np.sin(ang)])
        init_angles.append(0.0)
    # Second ring: 5 more hexes
    for k in range(5):
        ang = (k + 0.5) * np.pi / 3.0
        init_positions.append([2 * hex_dist * np.cos(ang), 2 * hex_dist * np.sin(ang)])
        init_angles.append(0.0)
    
    def objective(params):
        if time.time() > deadline - 5.0:
            return 1e10
        # params: [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12, cx, cy, S, A]
        # Total: 12*3 + 4 = 40
        inner_params = params[:36].reshape(12, 3)
        cx = params[36]
        cy = params[37]
        S = params[38]
        A = params[39]
        
        if S < 1.0:
            return 1e10
        
        # Compute outer hexagon vertices
        outer_verts = _outer_hex_vertices(cx, cy, S, A)
        
        total_penalty = 0.0
        inner_verts_list = []
        
        for i in range(12):
            xi, yi, ai = inner_params[i]
            iv = _hex_vertices(xi, yi, s_inner, ai)
            inner_verts_list.append(iv)
            
            # Check if inner hexagon is inside outer
            for iv_pt in iv:
                for j in range(6):
                    a = outer_verts[j]
                    b = outer_verts[(j + 1) % 6]
                    edge = b - a
                    to_p = iv_pt - a
                    cross = edge[0] * to_p[1] - edge[1] * to_p[0]
                    if cross < 0:
                        total_penalty += abs(cross) * 100.0
        
        # Check pairwise overlaps
        for i in range(12):
            for j in range(i + 1, 12):
                area = _poly_overlap_area(inner_verts_list[i], inner_verts_list[j])
                if area > 1e-8:
                    total_penalty += area * 1000.0
        
        return S + total_penalty
    
    x0 = np.array(init_positions + init_angles).flatten()
    x0 = np.concatenate([x0, [0.0, 0.0, 5.0, 0.0]])
    
    # Try multiple restarts with different initial configurations
    restarts = [
        # Original honeycomb
        None,
        # Slightly perturbed
        lambda: x0 + np.random.randn(40) * 0.1,
        # Different initial S
        lambda: np.concatenate([x0[:36], [0.0, 0.0, 4.5, 0.0]]),
        # Rotated initial
        lambda: np.concatenate([x0[:36], [0.0, 0.0, 5.0, 30.0]]),
    ]
    
    for restart_idx, restart in enumerate(restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart is None:
            x_init = x0.copy()
        else:
            x_init = restart()
        
        try:
            result = minimize(objective, x_init, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success or result.fun < best_side:
                # Verify the solution
                params = result.x
                inner_params = params[:36].reshape(12, 3)
                cx = params[36]
                cy = params[37]
                S = params[38]
                A = params[39]
                
                # Check validity
                outer_verts = _outer_hex_vertices(cx, cy, S, A)
                valid = True
                inner_verts_list = []
                for i in range(12):
                    xi, yi, ai = inner_params[i]
                    iv = _hex_vertices(xi, yi, s_inner, ai)
                    inner_verts_list.append(iv)
                    for iv_pt in iv:
                        for j in range(6):
                            a = outer_verts[j]
                            b = outer_verts[(j + 1) % 6]
                            edge = b - a
                            to_p = iv_pt - a
                            cross = edge[0] * to_p[1] - edge[1] * to_p[0]
                            if cross < -1e-4:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    for i in range(12):
                        for j in range(i + 1, 12):
                            area = _poly_overlap_area(inner_verts_list[i], inner_verts_list[j])
                            if area > 1e-4:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid and S < best_side:
                    best_side = S
                    best_result = (inner_params, (cx, cy), S, A)
        except Exception as e:
            pass
    
    if best_result is not None:
        inner_params, center, S, A = best_result
        inner = [[float(inner_params[i][0]), float(inner_params[i][1]), float(inner_params[i][2])] for i in range(12)]
        return inner, (float(center[0]), float(center[1])), float(S), float(A)
    
    # Fallback: return the initial guess with a reasonable outer size
    inner = [[float(init_positions[i][0]), float(init_positions[i][1]), 0.0] for i in range(12)]
    return inner, (0.0, 0.0), 6.0, 0.0
