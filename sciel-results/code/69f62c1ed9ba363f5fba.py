"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Combines the current program's Nelder-Mead refinement with the inspiration's
multi-topology lattice packing and faster geometry checks.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def hex_vertices_np(cx, cy, s, angle_rad):
    """Vectorized vertices of a regular hexagon."""
    thetas = angle_rad + np.arange(6) * (math.pi / 3.0)
    return np.column_stack([cx + s * np.cos(thetas), cy + s * np.sin(thetas)])

def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon (CCW or CW)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # normal to edge
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            # project A
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            # project B
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False  # separating axis found
    return True

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping.
    inner: list of (x, y, angle_deg) with unit side."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside.
    For a regular hexagon of side s, apothem = s * sqrt(3)/2.
    A point is inside iff for each of the 6 outward normals n_k, (p - c)·n_k <= apothem.
    So S >= (2/sqrt(3)) * max over all vertices and all k of (p-c)·n_k.
    """
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj

def compute_min_outer_side_fast(inner, outer_center, outer_angle_rad):
    """Vectorized computation of minimum outer side length.
    inner: list of (x, y, angle_deg) with unit side.
    outer_angle_rad: outer hexagon rotation in radians.
    Returns the minimum side length.
    """
    cx, cy = outer_center
    # 6 outward normals of the outer hexagon
    thetas = outer_angle_rad + (np.arange(6) + 0.5) * (math.pi / 3.0)
    normals = np.column_stack([np.cos(thetas), np.sin(thetas)])  # (6, 2)
    
    max_proj = 0.0
    for (x, y, ang) in inner:
        ang_rad = math.radians(ang)
        # vertices of this unit hexagon
        v_thetas = ang_rad + np.arange(6) * (math.pi / 3.0)
        vx = x + np.cos(v_thetas)
        vy = y + np.sin(v_thetas)
        # relative positions
        dx = vx - cx
        dy = vy - cy
        # project onto all 6 normals: (6,) dot (6,2) -> (6,)
        projs = dx @ normals[:, 0] + dy @ normals[:, 1]
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    return (2.0 / math.sqrt(3.0)) * max_proj

# --- Topology builders ---

def build_ring12(scale, inner_rot, outer_rot):
    """Ring of 12 hexagons around center."""
    inner = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append((x, y, inner_rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot):
    """Two concentric rings of 6, no center."""
    inner = []
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        x = scale1 * math.cos(theta)
        y = scale1 * math.sin(theta)
        inner.append((x, y, rot1))
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        x = scale2 * math.cos(theta)
        y = scale2 * math.sin(theta)
        inner.append((x, y, rot2))
    return inner, (0.0, 0.0), outer_rot

def build_rect3x4(scale_x, scale_y, rot, outer_rot):
    """3x4 rectangular grid."""
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex_lattice(rot, outer_rot):
    """Hexagonal lattice: 1 center + 6 ring + 5 more arranged."""
    inner = []
    inner.append((0.0, 0.0, rot))
    for k in range(6):
        theta = math.radians(rot + k * 60.0)
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        inner.append((x, y, rot))
    for k in range(5):
        theta = math.radians(rot + 30.0 + k * 60.0)
        x = 2.0 * math.sqrt(3) * math.cos(theta)
        y = 2.0 * math.sqrt(3) * math.sin(theta)
        inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def build_linear_chain(rot, outer_rot):
    """Linear chain of 12 hexagons."""
    inner = []
    for k in range(12):
        x = (k - 5.5) * 2.0
        y = 0.0
        inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def build_hexagonal_cluster(rot, outer_rot):
    """Compact hexagonal cluster: 1 center + 6 + 5."""
    inner = []
    inner.append((0.0, 0.0, rot))
    for k in range(6):
        theta = math.radians(rot + k * 60.0)
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        inner.append((x, y, rot))
    positions = [
        (2.0, 2.0 * math.sqrt(3)),
        (-2.0, 2.0 * math.sqrt(3)),
        (4.0, 0.0),
        (-4.0, 0.0),
        (2.0, -2.0 * math.sqrt(3)),
    ]
    for (px, py) in positions:
        inner.append((px, py, rot))
    return inner, (0.0, 0.0), outer_rot

def build_ring12_with_center(scale, inner_rot, outer_rot):
    """Ring of 12 hexagons + 1 center = 13? No, just 12 in ring."""
    return build_ring12(scale, inner_rot, outer_rot)

def build_two_rings_6_6_offset(scale1, scale2, rot1, rot2, offset_deg, outer_rot):
    """Two concentric rings of 6 with angular offset between them."""
    inner = []
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        x = scale1 * math.cos(theta)
        y = scale1 * math.sin(theta)
        inner.append((x, y, rot1))
    for k in range(6):
        theta = math.radians(rot2 + offset_deg + k * 60.0)
        x = scale2 * math.cos(theta)
        y = scale2 * math.sin(theta)
        inner.append((x, y, rot2))
    return inner, (0.0, 0.0), outer_rot

# --- Optimization ---

def optimize_topology(topology, x0, bounds, outer_rot, time_budget):
    """Optimize a single topology using Nelder-Mead."""
    start_time = time.time()
    
    def objective(params):
        if topology == 'ring12':
            scale, inner_rot = params
            if scale < 0.5 or scale > 3.0:
                return 1000.0
            inner, center, orot = build_ring12(scale, inner_rot, outer_rot)
        elif topology == 'hex6+6':
            scale1, scale2, rot1, rot2 = params
            if scale1 < 0.5 or scale2 < 0.5 or scale2 < scale1 + 0.3:
                return 1000.0
            inner, center, orot = build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot)
        elif topology == 'rect3x4':
            sx, sy, rot = params
            if sx < 1.5 or sy < 1.5:
                return 1000.0
            inner, center, orot = build_rect3x4(sx, sy, rot, outer_rot)
        elif topology == 'hex_lattice':
            rot = params[0]
            inner, center, orot = build_hex_lattice(rot, outer_rot)
        elif topology == 'linear_chain':
            rot = params[0]
            inner, center, orot = build_linear_chain(rot, outer_rot)
        elif topology == 'hexagonal_cluster':
            rot = params[0]
            inner, center, orot = build_hexagonal_cluster(rot, outer_rot)
        else:
            return 1000.0
        
        side = compute_min_outer_side(inner, center, orot)
        if not check_validity(inner, center, side, orot):
            return side + 10.0  # penalty for invalid
        return side
    
    best_x = x0
    best_f = objective(x0)
    
    # Coarse grid search
    n_grid = 6
    if len(x0) == 2:
        grid1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        grid2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
        for g1 in grid1:
            for g2 in grid2:
                f = objective([g1, g2])
                if f < best_f:
                    best_f = f
                    best_x = [g1, g2]
                if time.time() - start_time > time_budget:
                    return best_x, best_f
    elif len(x0) == 3:
        grid1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        grid2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
        grid3 = np.linspace(bounds[2][0], bounds[2][1], n_grid)
        for g1 in grid1:
            for g2 in grid2:
                for g3 in grid3:
                    f = objective([g1, g2, g3])
                    if f < best_f:
                        best_f = f
                        best_x = [g1, g2, g3]
                    if time.time() - start_time > time_budget:
                        return best_x, best_f
    elif len(x0) == 4:
        grid1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        grid2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
        grid3 = np.linspace(bounds[2][0], bounds[2][1], n_grid)
        grid4 = np.linspace(bounds[3][0], bounds[3][1], n_grid)
        for g1 in grid1:
            for g2 in grid2:
                if g2 < g1 + 0.3:
                    continue
                for g3 in grid3:
                    for g4 in grid4:
                        f = objective([g1, g2, g3, g4])
                        if f < best_f:
                            best_f = f
                            best_x = [g1, g2, g3, g4]
                        if time.time() - start_time > time_budget:
                            return best_x, best_f
    elif len(x0) == 1:
        grid1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        for g1 in grid1:
            f = objective([g1])
            if f < best_f:
                best_f = f
                best_x = [g1]
            if time.time() - start_time > time_budget:
                return best_x, best_f
    
    # Nelder-Mead refinement
    try:
        result = minimize(
            objective,
            best_x,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'maxfev': 5000,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        if result.fun < best_f:
            best_f = result.fun
            best_x = result.x
    except Exception:
        pass
    
    return best_x, best_f

def fine_rotation_sweep(topology, base_params, outer_rot_range, n_outer, time_budget, start_time):
    """Sweep outer rotation finely for a given topology and base parameters.
    Uses vectorized support function for speed.
    """
    best_side = float('inf')
    best_orot = 0.0
    
    outer_rots = np.linspace(outer_rot_range[0], outer_rot_range[1], n_outer)
    
    for orot_deg in outer_rots:
        if time.time() - start_time > time_budget:
            break
        orot_rad = math.radians(orot_deg)
        # Build the inner configuration
        if topology == 'ring12':
            scale, inner_rot = base_params
            inner, center, _ = build_ring12(scale, inner_rot, 0.0)
        elif topology == 'hex6+6':
            scale1, scale2, rot1, rot2 = base_params
            inner, center, _ = build_hex6_plus6(scale1, scale2, rot1, rot2, 0.0)
        elif topology == 'rect3x4':
            sx, sy, rot = base_params
            inner, center, _ = build_rect3x4(sx, sy, rot, 0.0)
        elif topology == 'hex_lattice':
            rot = base_params[0]
            inner, center, _ = build_hex_lattice(rot, 0.0)
        elif topology == 'linear_chain':
            rot = base_params[0]
            inner, center, _ = build_linear_chain(rot, 0.0)
        elif topology == 'hexagonal_cluster':
            rot = base_params[0]
            inner, center, _ = build_hexagonal_cluster(rot, 0.0)
        else:
            continue
        
        side = compute_min_outer_side_fast(inner, center, orot_rad)
        if side < best_side:
            best_side = side
            best_orot = orot_deg
    
    return best_side, best_orot

def polish_result(inner, center, outer_angle, time_budget, start_time):
    """High-precision polish: locally adjust outer rotation and center to minimize side."""
    best_side = compute_min_outer_side_fast(inner, center, math.radians(outer_angle))
    best_orot = outer_angle
    best_center = center
    
    # Fine sweep of outer rotation around current best
    for orot_deg in np.linspace(outer_angle - 5, outer_angle + 5, 41):
        if time.time() - start_time > time_budget:
            break
        side = compute_min_outer_side_fast(inner, center, math.radians(orot_deg))
        if side < best_side:
            best_side = side
            best_orot = orot_deg
    
    # Even finer sweep
    for orot_deg in np.linspace(best_orot - 1, best_orot + 1, 21):
        if time.time() - start_time > time_budget:
            break
        side = compute_min_outer_side_fast(inner, center, math.radians(orot_deg))
        if side < best_side:
            best_side = side
            best_orot = orot_deg
    
    # Try small center shifts
    for dx in np.linspace(-0.1, 0.1, 11):
        for dy in np.linspace(-0.1, 0.1, 11):
            if time.time() - start_time > time_budget:
                break
            new_center = (center[0] + dx, center[1] + dy)
            side = compute_min_outer_side_fast(inner, new_center, math.radians(best_orot))
            if side < best_side:
                best_side = side
                best_center = new_center
    
    return best_side, best_orot, best_center

def construct():
    start_time = time.time()
    
    # Define topologies to try
    topologies = [
        ('ring12', [2.0, 0.0], [(0.8, 2.5), (0, 360)]),
        ('hex6+6', [1.5, 3.0, 0.0, 30.0], [(0.8, 2.5), (1.5, 4.0), (0, 360), (0, 360)]),
        ('rect3x4', [2.0, 2.0, 0.0], [(1.5, 3.0), (1.5, 3.0), (0, 360)]),
        ('hex_lattice', [0.0], [(0, 360)]),
        ('linear_chain', [0.0], [(0, 360)]),
        ('hexagonal_cluster', [0.0], [(0, 360)]),
    ]
    
    best_result = None
    best_side = float('inf')
    
    # Phase 1: Coarse topology optimization
    for topology, x0, bounds in topologies:
        if time.time() > DEADLINE:
            break
        
        remaining = DEADLINE - time.time()
        time_budget = min(12.0, remaining * 0.25)
        
        x_opt, f_opt = optimize_topology(topology, x0, bounds, 0.0, time_budget)
        
        if f_opt < best_side:
            best_side = f_opt
            best_result = (topology, x_opt)
    
    # Phase 2: Fine rotation sweep for top candidates
    if time.time() < DEADLINE - 20:
        # Re-optimize top 3 topologies with fine outer rotation sweep
        for topology, x0, bounds in topologies:
            if time.time() > DEADLINE - 15:
                break
            
            # Quick evaluation at default params
            if topology == 'ring12':
                inner, center, _ = build_ring12(x0[0], x0[1], 0.0)
            elif topology == 'hex6+6':
                inner, center, _ = build_hex6_plus6(x0[0], x0[1], x0[2], x0[3], 0.0)
            elif topology == 'rect3x4':
                inner, center, _ = build_rect3x4(x0[0], x0[1], x0[2], 0.0)
            elif topology == 'hex_lattice':
                inner, center, _ = build_hex_lattice(x0[0], 0.0)
            elif topology == 'linear_chain':
                inner, center, _ = build_linear_chain(x0[0], 0.0)
            elif topology == 'hexagonal_cluster':
                inner, center, _ = build_hexagonal_cluster(x0[0], 0.0)
            else:
                continue
            
            # Fine sweep over outer rotation (0 to 30 degrees, 60-fold symmetry)
            sweep_budget = min(8.0, DEADLINE - time.time() - 10)
            if sweep_budget < 2:
                break
            
            side, orot = fine_rotation_sweep(
                topology, x0, (0, 30), 61, sweep_budget, time.time()
            )
            
            if side < best_side:
                best_side = side
                best_result = (topology, x0, orot)
                # Store with outer rotation
                if len(best_result) == 2:
                    best_result = (topology, x0, orot)
    
    # Phase 3: If we have a good candidate, do a polish pass
    if best_result is not None and time.time() < DEADLINE - 5:
        topology = best_result[0]
        if len(best_result) == 3:
            params, orot = best_result[1], best_result[2]
        else:
            params, orot = best_result[1], 0.0
        
        if topology == 'ring12':
            inner, center, _ = build_ring12(params[0], params[1], 0.0)
        elif topology == 'hex6+6':
            inner, center, _ = build_hex6_plus6(params[0], params[1], params[2], params[3], 0.0)
        elif topology == 'rect3x4':
            inner, center, _ = build_rect3x4(params[0], params[1], params[2], 0.0)
        elif topology == 'hex_lattice':
            inner, center, _ = build_hex_lattice(params[0], 0.0)
        elif topology == 'linear_chain':
            inner, center, _ = build_linear_chain(params[0], 0.0)
        elif topology == 'hexagonal_cluster':
            inner, center, _ = build_hexagonal_cluster(params[0], 0.0)
        else:
            inner, center, _ = build_ring12(2.0, 0.0, 0.0)
        
        polish_budget = min(10.0, DEADLINE - time.time() - 2)
        if polish_budget > 1:
            final_side, final_orot, final_center = polish_result(
                inner, center, orot, polish_budget, time.time()
            )
        else:
            final_side = compute_min_outer_side_fast(inner, center, math.radians(orot))
            final_orot = orot
            final_center = center
        
        # Verify
        if not check_validity(inner, final_center, final_side, final_orot):
            # Fallback
            grid_xs = [-2.0, 0.0, 2.0]
            grid_ys = [0.0, math.sqrt(3), 2*math.sqrt(3), 3*math.sqrt(3)]
            positions = []
            for y in grid_ys:
                for x in grid_xs:
                    positions.append((x, y, 0.0))
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            inner = [(px - cx, py - cy, angle) for px, py, angle in positions]
            final_center = (0.0, 0.0)
            final_orot = 0.0
            final_side = compute_min_outer_side(inner, final_center, final_orot)
    else:
        # Fallback: 3x4 grid
        grid_xs = [-2.0, 0.0, 2.0]
        grid_ys = [0.0, math.sqrt(3), 2*math.sqrt(3), 3*math.sqrt(3)]
        positions = []
        for y in grid_ys:
            for x in grid_xs:
                positions.append((x, y, 0.0))
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        inner = [(px - cx, py - cy, angle) for px, py, angle in positions]
        final_center = (0.0, 0.0)
        final_orot = 0.0
        final_side = compute_min_outer_side(inner, final_center, final_orot)
    
    inner_list = []
    for x, y, angle in inner:
        inner_list.append([float(x), float(y), float(angle)])
    
    return inner_list, (float(final_center[0]), float(final_center[1])), float(final_side), float(final_orot)
