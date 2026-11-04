"""
Packing of 12 unit hexagons in a minimal outer hexagon.
Uses a robust geometric construction with local refinement.
"""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 110.0  # Safety margin below 120s
INRADIUS = math.sqrt(3) / 2.0

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _check_overlap(verts_a, verts_b):
    axes_a = _axes_from_polygon(verts_a)
    axes_b = _axes_from_polygon(verts_b)
    for axis in axes_a + axes_b:
        min_a, max_a = _project_polygon(verts_a, axis)
        min_b, max_b = _project_polygon(verts_b, axis)
        if max_a < min_b or max_b < min_a:
            return False
    return True

def _is_valid(cfg):
    """Check if configuration has no overlaps."""
    verts_list = []
    for p in cfg:
        verts_list.append(_hex_vertices(p[0], p[1], p[2]))
    
    for i in range(N):
        for j in range(i + 1, N):
            # Quick distance check
            dx = cfg[i][0] - cfg[j][0]
            dy = cfg[i][1] - cfg[j][1]
            if dx*dx + dy*dy < (2 * INRADIUS - 0.1)**2: # Tight bound
                if _check_overlap(verts_list[i], verts_list[j]):
                    return False
    return True

def _compute_outer_side(cfg, center, angle_deg):
    """Calculate the side length of the smallest enclosing hexagon."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for p in cfg:
        verts = _hex_vertices(p[0], p[1], p[2])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _find_best_outer(cfg):
    """Find optimal outer hexagon center and rotation."""
    all_verts = []
    for p in cfg:
        all_verts.extend(_hex_vertices(p[0], p[1], p[2]))
    
    cx0 = np.mean([v[0] for v in all_verts])
    cy0 = np.mean([v[1] for v in all_verts])
    
    best_side = float('inf')
    best_cx, best_cy, best_angle = cx0, cy0, 0.0
    
    # Coarse search
    for angle_deg in range(0, 60, 15):
        for dx in np.linspace(-1, 1, 5):
            for dy in np.linspace(-1, 1, 5):
                cx, cy = cx0 + dx, cy0 + dy
                side = _compute_outer_side(cfg, (cx, cy), angle_deg)
                if side < best_side:
                    best_side = side
                    best_cx, best_cy, best_angle = cx, cy, angle_deg
    
    # Fine search
    for angle_deg in np.arange(best_angle - 15, best_angle + 15, 1.0):
        for dx in np.linspace(-0.5, 0.5, 5):
            for dy in np.linspace(-0.5, 0.5, 5):
                cx, cy = cx0 + dx, cy0 + dy
                side = _compute_outer_side(cfg, (cx, cy), angle_deg)
                if side < best_side:
                    best_side = side
                    best_cx, best_cy, best_angle = cx, cy, angle_deg
    
    return best_cx, best_cy, best_side, best_angle

def construct():
    t0 = time.time()
    random.seed(42)
    
    # 1. Initial Topology: "Flower" + 5 extensions
    # Center at (0,0)
    # 6 neighbors at distance 2*sqrt(3)
    # 5 more attached to outer edges
    d = 2 * math.sqrt(3)
    cfg = []
    cfg.append([0.0, 0.0, 0.0])
    
    for i in range(6):
        ang = math.radians(60 * i)
        x = d * math.cos(ang)
        y = d * math.sin(ang)
        cfg.append([x, y, 0.0])
        
        # Attach one more hexagon to the "outer" face of this neighbor
        # The face normal is in the same direction as the vector from center
        # Distance from center to this new hex is d + d = 2d? No, side length 1 hex.
        # Distance between centers of touching hexes is 2*sqrt(3).
        # So new center is at distance d + d = 2d from center?
        # Wait, the neighbor is at distance d. The new hex touches the neighbor's outer face.
        # So distance from center to new hex is 2*d.
        x2 = 2 * d * math.cos(ang)
        y2 = 2 * d * math.sin(ang)
        cfg.append([x2, y2, 0.0])
        
    # We have 1 + 6 + 6 = 13 hexagons. We need 12.
    # Remove one of the outer ones to make it 12? Or adjust topology.
    # Let's keep 1 center, 6 ring, and 5 outer.
    # So remove the last one added in the loop?
    cfg = cfg[:12]
    
    # Validate initial
    if not _is_valid(cfg):
        # Fallback to simple grid if construction fails (unlikely for this topology)
        cfg = []
        for i in range(4):
            for j in range(3):
                cfg.append([i * 2 * math.sqrt(3) - 3 * math.sqrt(3), j * 2 - 2, 0.0])
        # Scale up to avoid overlaps if any
        if not _is_valid(cfg):
             for p in cfg:
                 p[0] *= 1.5
                 p[1] *= 1.5
                 
    best_cfg = [p[:] for p in cfg]
    cx, cy, best_side, best_angle = _find_best_outer(best_cfg)
    
    # 2. Local Refinement (Hill Climbing)
    # Try to minimize side length by moving hexagons slightly
    step_size = 0.1
    improvements = 0
    max_improvements = 50
    
    while improvements < max_improvements and time.time() - t0 < TIME_LIMIT - 10:
        improved = False
        for i in range(N):
            # Try small perturbations
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    new_cfg = [p[:] for p in best_cfg]
                    new_cfg[i][0] += dx
                    new_cfg[i][1] += dy
                    
                    if _is_valid(new_cfg):
                        c2, c3, s2, a2 = _find_best_outer(new_cfg)
                        if s2 < best_side - 1e-5:
                            best_cfg = new_cfg
                            cx, cy, best_side, best_angle = c2, c3, s2, a2
                            improvements += 1
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try rotation perturbations
            for i in range(N):
                for da in [-step_size, step_size]:
                    new_cfg = [p[:] for p in best_cfg]
                    new_cfg[i][2] += da
                    if _is_valid(new_cfg):
                        c2, c3, s2, a2 = _find_best_outer(new_cfg)
                        if s2 < best_side - 1e-5:
                            best_cfg = new_cfg
                            cx, cy, best_side, best_angle = c2, c3, s2, a2
                            improvements += 1
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break # No improvement found
        
    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, (cx, cy), best_side, best_angle
