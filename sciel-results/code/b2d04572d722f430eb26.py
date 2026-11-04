import numpy as np
import time
import math
import random

N = 12
DEADLINE = 142.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _sat_overlap(v1, v2):
    """
    Returns penetration depth if overlapping (negative), or separation distance if not.
    Uses Separating Axis Theorem for convex polygons.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)

    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)

        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated

        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)

    return -min_overlap  # negative = overlapping

def _outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    # Edge normals for a regular hexagon with vertex at angle 0 are at 30, 90, 150, 210, 270, 330
    # But wait, the vertices are at 0, 60, ...
    # The edges are between vertices.
    # Normal to edge between v0 and v1 is at angle 30.
    # So normals are at 30, 90, 150, 210, 270, 330.
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Project points onto normals
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    
    # For a regular hexagon with side S, the apothem is S * sqrt(3) / 2.
    # The distance from center to edge is S * sqrt(3) / 2.
    # So max_proj <= S * sqrt(3) / 2  =>  S >= 2 * max_proj / sqrt(3)
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner):
    """Check no two inner hexagons overlap."""
    n = len(inner)
    verts = [_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap(verts[i], verts[j])
            if d < -1e-9:
                return False
    return True

def _all_inside(inner, outer_center, outer_side, outer_angle):
    """Check all inner hexagon vertices are inside outer hexagon."""
    # Shift inner points relative to outer center
    pts = []
    for (cx, cy, ang) in inner:
        v = _hex_vertices(cx, cy, 1.0, ang)
        v[:, 0] -= outer_center[0]
        v[:, 1] -= outer_center[1]
        pts.append(v)
    all_pts = np.vstack(pts)
    
    # Calculate required side for this configuration
    req_side = _outer_hex_side(all_pts, outer_angle)
    return req_side <= outer_side + 1e-9

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)

    def objective(params):
        if time.time() - t0 > DEADLINE - 5:
            return 1e6
        
        # Unpack parameters
        # Structure: [x0, y0, r0, x1, y1, r1, ..., x11, y11, r11, outer_rot]
        # But it's easier to handle outer center too? 
        # Actually, due to symmetry, we can fix outer center at (0,0) without loss of generality 
        # if we allow inner positions to be anywhere. 
        # However, allowing outer center to move might help if the inner cluster is not centered?
        # No, if we fix outer center at (0,0), we can just center the inner cluster around (0,0) 
        # or let the optimizer find the best position relative to (0,0).
        # Let's include outer center (ox, oy) and outer rotation (oang) in the parameters to be safe.
        # Total vars: 12*3 + 3 = 39
        
        inner_params = params[:36].reshape(12, 3)
        ox = params[36]
        oy = params[37]
        oang = params[38]
        
        verts_list = []
        for i in range(12):
            cx, cy, rot = inner_params[i]
            v = _hex_vertices(cx, cy, 1.0, rot)
            verts_list.append(v)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                d = _sat_overlap(verts_list[i], verts_list[j])
                if d < 0:
                    penalty += (-d) ** 2 * 10.0
        
        # Calculate outer side needed
        # Shift points by outer center
        all_pts = np.vstack(verts_list)
        all_pts[:, 0] -= ox
        all_pts[:, 1] -= oy
        
        L = _outer_hex_side(all_pts, oang)
        
        return L + penalty

    # Generate initial configurations
    configs = []
    
    # 1. Honeycomb grid 4x3 (or 3x4)
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        # For pointy-top (vertex at 90? No, our hex has vertex at 0, so it's "flat" top/bottom? 
        # Vertex at 0 means (1,0). So it's wide horizontally.
        # Width = 2, Height = sqrt(3).
        # Spacing x: 2 (touching)
        # Spacing y: sqrt(3) (touching)
        # Offset alternate rows by 1.
        spacing_x = 2.0 + gap
        spacing_y = np.sqrt(3) + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        # Center
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        return pts

    # 2. Ring configuration (1 center, 6 ring, 5 outer)
    def make_ring(gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        # Distance for touching hexes in ring: sqrt(3)
        d1 = np.sqrt(3) + gap
        d2 = 2 * np.sqrt(3) + gap
        
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        
        pts = np.array(pts)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        return pts

    # 3. Linear strip
    def make_strip(n, gap=0.0):
        pts = []
        for i in range(n):
            pts.append([i * (2.0 + gap), 0, 0.0])
        pts = np.array(pts)
        pts[:, 0] -= np.mean(pts[:, 0])
        return pts

    gap = 0.1
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    
    # Rotated variants
    for base in [make_hex_grid(4, 3, gap=gap), make_hex_grid(3, 4, gap=gap)]:
        rotated = base.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
        
    # Add random configurations
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.randn(12, 3) * 2.0
        pts[:, 2] = np.random.choice([0, 30, 60, 90, 120, 150], size=12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)

    best_L = float('inf')
    best_params = None

    outer_rots = [0.0, 30.0, 15.0, 45.0]

    for cfg in configs:
        if time_left() < 15:
            break
            
        for orot in outer_rots:
            if time_left() < 5:
                break
                
            init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, orot]))
            
            # Evaluate initial
            val = objective(init_params)
            # Check validity
            inner_init = init_params[:36].reshape(12, 3)
            verts_init = [_hex_vertices(inner_init[i, 0], inner_init[i, 1], 1.0, inner_init[i, 2]) for i in range(12)]
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    if _sat_overlap(verts_init[i], verts_init[j]) < -1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                L_val = objective(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Optimize
            try:
                from scipy.optimize import minimize
                res = minimize(
                    objective, 
                    init_params, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                
                # Check result validity
                res_inner = res.x[:36].reshape(12, 3)
                res_ox = res.x[36]
                res_oy = res.x[37]
                res_oang = res.x[38]
                
                verts_res = [_hex_vertices(res_inner[i, 0], res_inner[i, 1], 1.0, res_inner[i, 2]) for i in range(12)]
                valid = True
                for i in range(12):
                    for j in range(i + 1, 12):
                        if _sat_overlap(verts_res[i], verts_res[j]) < -1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    all_pts = np.vstack(verts_res)
                    all_pts[:, 0] -= res_ox
                    all_pts[:, 1] -= res_oy
                    L_res = _outer_hex_side(all_pts, res_oang)
                    
                    if L_res < best_L:
                        best_L = L_res
                        best_params = res.x.copy()
                        
            except Exception:
                pass

    if best_params is None:
        # Fallback to a simple valid configuration
        cfg = make_hex_grid(4, 3, gap=0.1)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        best_L = objective(best_params)

    # Extract final solution
    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
    # Final verification and adjustment
    if not _check_disjoint(inner):
        # Fallback
        cfg = make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side(all_pts, outer_angle)
    
    # Add tiny epsilon to ensure strict containment if needed
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle
