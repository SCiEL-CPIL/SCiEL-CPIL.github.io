"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses multi-topology lattice packing with smooth epigraph objective and local refinement.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

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

def compute_min_outer_side_smooth(inner, outer_center, outer_angle, beta=50.0):
    """Smooth approximation of compute_min_outer_side using log-sum-exp.
    Replaces max with soft-max: (1/beta)*log(sum(exp(beta*x_i))).
    """
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Collect all projections
    projs = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                projs.append(dx * nx + dy * ny)
    
    if not projs:
        return 0.0
    
    arr = np.array(projs)
    max_val = arr.max()
    if max_val <= 0:
        return 0.0
    # log-sum-exp for numerical stability
    soft_max = max_val + math.log(np.sum(np.exp(beta * (arr - max_val)))) / beta
    return (2.0 / math.sqrt(3.0)) * soft_max

def polygons_min_distance(A, B):
    """Minimum distance between two convex polygons using SAT.
    Returns 0 if overlapping, positive distance if separated.
    """
    min_dist = float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-15:
                continue
            nx, ny = -dy / length, dx / length
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            gap = min(minA - maxB, minB - maxA)
            if gap > 0:
                min_dist = min(min_dist, gap)
            else:
                return 0.0
    return min_dist

def compute_overlap_penalty(inner, beta=20.0):
    """Smooth penalty for pairwise overlaps using soft-min of distances."""
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    
    penalty = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            dist = polygons_min_distance(polys[i], polys[j])
            if dist < 0.5:
                # Soft penalty: exp(-beta * dist) when dist < 0
                penalty += math.exp(-beta * dist)
    return penalty

def compute_containment_penalty(inner, outer_center, outer_side, outer_angle, beta=20.0):
    """Smooth penalty for inner hexagons not fully inside outer."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    apothem = outer_side * math.sqrt(3) / 2.0
    
    penalty = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - outer_center[0], py - outer_center[1]
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > apothem:
                    penalty += math.exp(-beta * (apothem - proj))
    return penalty

def smooth_objective(params, topology, outer_rot, beta_side=50.0, beta_overlap=20.0, beta_contain=20.0):
    """Smooth objective combining side length, overlap penalty, and containment penalty."""
    if topology == 'ring12':
        scale, inner_rot = params
        if scale < 0.5 or scale > 3.5:
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
    elif topology == 'cluster12':
        # 12 hexagons in a compact cluster: center + 6 ring + 5 more
        cx, cy, rot = params
        inner = [(cx, cy, rot)]
        # First ring of 6
        for k in range(6):
            theta = math.radians(rot + k * 60.0)
            x = cx + 2.0 * math.cos(theta)
            y = cy + 2.0 * math.sin(theta)
            inner.append((x, y, rot))
        # Second partial ring of 5
        for k in range(5):
            theta = math.radians(rot + 30.0 + k * 60.0)
            x = cx + 3.464 * math.cos(theta)  # 2*sqrt(3)
            y = cy + 3.464 * math.sin(theta)
            inner.append((x, y, rot))
    elif topology == 'linear':
        # Linear chain of 12
        cx, cy, rot = params
        spacing = 2.0
        inner = []
        for k in range(12):
            x = cx + (k - 5.5) * spacing
            y = cy
            inner.append((x, y, rot))
    else:
        return 1000.0
    
    side = compute_min_outer_side_smooth(inner, center, orot, beta_side)
    overlap_pen = compute_overlap_penalty(inner, beta_overlap)
    # For containment, we need a reference outer size; use current computed side + margin
    contain_pen = compute_containment_penalty(inner, center, side + 0.1, orot, beta_contain)
    
    return side + 10.0 * overlap_pen + 5.0 * contain_pen

# --- Multi-topology lattice packing ---

def build_ring12(scale, inner_rot, outer_rot):
    """Ring of 12 hexagons around center."""
    inner = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append((x, y, inner_rot))
    return inner, (0.0, 0.0), outer_rot

def build_cluster12(cx, cy, rot, outer_rot):
    """Compact cluster: 1 center + 6 inner ring + 5 outer ring."""
    inner = [(cx, cy, rot)]
    for k in range(6):
        theta = math.radians(rot + k * 60.0)
        x = cx + 2.0 * math.cos(theta)
        y = cy + 2.0 * math.sin(theta)
        inner.append((x, y, rot))
    for k in range(5):
        theta = math.radians(rot + 30.0 + k * 60.0)
        x = cx + 3.464 * math.cos(theta)
        y = cy + 3.464 * math.sin(theta)
        inner.append((x, y, rot))
    return inner, (cx, cy), outer_rot

def build_linear(cx, cy, rot, outer_rot):
    """Linear chain of 12 hexagons."""
    inner = []
    spacing = 2.0
    for k in range(12):
        x = cx + (k - 5.5) * spacing
        y = cy
        inner.append((x, y, rot))
    return inner, (cx, cy), outer_rot

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

def objective(params, topology, outer_rot):
    """Objective function for optimization (exact, with validity check)."""
    if topology == 'ring12':
        scale, inner_rot = params
        if scale < 0.5 or scale > 3.5:
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
    elif topology == 'cluster12':
        cx, cy, rot = params
        inner, center, orot = build_cluster12(cx, cy, rot, outer_rot)
    elif topology == 'linear':
        cx, cy, rot = params
        inner, center, orot = build_linear(cx, cy, rot, outer_rot)
    else:
        return 1000.0
    
    side = compute_min_outer_side(inner, center, orot)
    if not check_validity(inner, center, side, orot):
        return side + 10.0  # penalty for invalid
    return side

def optimize_topology(topology, outer_rot, time_budget):
    """Optimize a single topology using smooth SLSQP then exact polish."""
    start_time = time.time()
    
    if topology == 'ring12':
        x0 = [2.0, 0.0]
        bounds = [(0.8, 3.0), (-180, 180)]
        x0_list = [[2.0, 0.0], [1.8, 30.0], [2.2, -30.0], [1.6, 15.0]]
    elif topology == 'hex6+6':
        x0 = [1.5, 3.0, 0.0, 30.0]
        bounds = [(0.8, 3.0), (1.5, 4.5), (-180, 180), (-180, 180)]
        x0_list = [[1.5, 3.0, 0.0, 30.0], [1.8, 3.2, 15.0, 45.0], [1.3, 2.8, -15.0, 15.0]]
    elif topology == 'rect3x4':
        x0 = [2.0, 2.0, 0.0]
        bounds = [(1.5, 3.5), (1.5, 3.5), (-180, 180)]
        x0_list = [[2.0, 2.0, 0.0], [2.2, 1.8, 30.0], [1.8, 2.2, -30.0]]
    elif topology == 'cluster12':
        x0 = [0.0, 0.0, 0.0]
        bounds = [(-2.0, 2.0), (-2.0, 2.0), (-180, 180)]
        x0_list = [[0.0, 0.0, 0.0], [0.5, 0.5, 30.0], [-0.5, -0.5, -30.0]]
    elif topology == 'linear':
        x0 = [0.0, 0.0, 0.0]
        bounds = [(-5.0, 5.0), (-5.0, 5.0), (-180, 180)]
        x0_list = [[0.0, 0.0, 0.0], [0.0, 0.0, 90.0]]
    else:
        return None, None, None
    
    best_x = x0
    best_f = objective(x0, topology, outer_rot)
    
    # Evaluate all initial points
    for x_init in x0_list:
        f_init = objective(x_init, topology, outer_rot)
        if f_init < best_f:
            best_f = f_init
            best_x = x_init
    
    # Phase 1: Smooth SLSQP optimization
    if time.time() - start_time < time_budget * 0.6:
        try:
            result = minimize(smooth_objective, best_x, args=(topology, outer_rot),
                             method='SLSQP', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-8})
            if result.fun < best_f + 5.0:  # Allow some slack for smooth vs exact
                # Verify with exact objective
                f_exact = objective(result.x, topology, outer_rot)
                if f_exact < best_f:
                    best_x = result.x
                    best_f = f_exact
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead local refinement on exact objective
    if time.time() - start_time < time_budget * 0.85:
        try:
            result = minimize(objective, best_x, args=(topology, outer_rot),
                             method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if time.time() - start_time < time_budget * 0.95:
        try:
            result = minimize(objective, best_x, args=(topology, outer_rot),
                             method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    return best_x, best_f, topology

def optimize():
    """Try all topologies and outer rotations."""
    best_side = 40.0
    best_params = None
    best_topology = None
    best_outer_rot = 0.0
    best_valid = False
    
    topologies = ['ring12', 'hex6+6', 'rect3x4', 'cluster12', 'linear']
    outer_rots = [0.0, 30.0, 60.0, 15.0]
    
    # Allocate time: prioritize promising topologies
    total_time = max(1.0, DEADLINE - time.time())
    n_combos = len(topologies) * len(outer_rots)
    time_per = total_time / n_combos * 0.9  # 90% for search, 10% margin
    
    for orot in outer_rots:
        for topo in topologies:
            if time.time() > DEADLINE:
                break
            params, f, topo_used = optimize_topology(topo, orot, time_per)
            if params is not None and f < best_side:
                # Verify validity
                if topo_used == 'ring12':
                    inner, center, orot_used = build_ring12(params[0], params[1], orot)
                elif topo_used == 'hex6+6':
                    inner, center, orot_used = build_hex6_plus6(params[0], params[1], params[2], params[3], orot)
                elif topo_used == 'rect3x4':
                    inner, center, orot_used = build_rect3x4(params[0], params[1], params[2], orot)
                elif topo_used == 'cluster12':
                    inner, center, orot_used = build_cluster12(params[0], params[1], params[2], orot)
                elif topo_used == 'linear':
                    inner, center, orot_used = build_linear(params[0], params[1], params[2], orot)
                else:
                    continue
                
                side = compute_min_outer_side(inner, center, orot_used)
                if check_validity(inner, center, side, orot_used):
                    best_side = side
                    best_params = params
                    best_topology = topo_used
                    best_outer_rot = orot
                    best_valid = True
    
    return best_side, best_params, best_topology, best_outer_rot, best_valid

def construct():
    t0 = time.time()
    try:
        side, params, topology, outer_rot, valid = optimize()
        if valid and params is not None:
            if topology == 'ring12':
                inner, center, orot = build_ring12(params[0], params[1], outer_rot)
            elif topology == 'hex6+6':
                inner, center, orot = build_hex6_plus6(params[0], params[1], params[2], params[3], outer_rot)
            elif topology == 'rect3x4':
                inner, center, orot = build_rect3x4(params[0], params[1], params[2], outer_rot)
            elif topology == 'cluster12':
                inner, center, orot = build_cluster12(params[0], params[1], params[2], outer_rot)
            elif topology == 'linear':
                inner, center, orot = build_linear(params[0], params[1], params[2], outer_rot)
            else:
                raise ValueError("Unknown topology")
            return inner, center, side, orot
    except Exception:
        pass

    # Fallback: compact hexagonal arrangement (always valid)
    # 1 center + 6 ring + 5 outer = 12 hexagons
    inner = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        inner.append((x, y, 0.0))
    for k in range(5):
        theta = math.radians(30.0 + k * 60.0)
        x = 3.464 * math.cos(theta)
        y = 3.464 * math.sin(theta)
        inner.append((x, y, 0.0))
    # Compute minimum enclosing hexagon side
    side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
    # Verify and adjust if needed
    if not check_validity(inner, (0.0, 0.0), side, 0.0):
        side += 0.1
    return inner, (0.0, 0.0), side, 0.0
