"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses differential evolution with smooth penalty formulation for global search.
"""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def smooth_max(values, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    m = max(values)
    if m == 0:
        return 0.0
    exps = [math.exp(beta * (v - m)) for v in values]
    s = sum(exps)
    if s == 0:
        return m
    return m + math.log(s) / beta


def polygon_overlap_depth(A, B):
    """
    Compute the minimum overlap depth between two convex polygons using SAT.
    Returns a positive value if they overlap (penetration depth), 0 if separated or touching.
    """
    min_overlap = 0.0
    for poly in (A, B):
        m = len(poly)
        for k in range(m):
            x1, y1 = poly[k]
            x2, y2 = poly[(k + 1) % m]
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-12:
                continue
            nx, ny = -dy / length, dx / length
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA >= minB and maxB >= minA:
                # Overlapping on this axis; compute penetration
                # The overlap amount is min(maxA - minB, maxB - minA)
                overlap = min(maxA - minB, maxB - minA)
                if overlap > min_overlap:
                    min_overlap = overlap
    return min_overlap


def smooth_overlap_penalty(polys, beta=50.0):
    """
    Smooth penalty for pairwise overlaps.
    Returns 0 when all pairs are separated, positive when overlapping.
    Uses the overlap depth from SAT and applies a smooth penalty.
    """
    penalty = 0.0
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            depth = polygon_overlap_depth(polys[i], polys[j])
            if depth > 1e-9:
                # Smooth penalty: use softplus-like function
                # penalty = depth * (1 - exp(-beta * depth)) approximates depth for large depth
                # but is smoother near 0
                penalty += depth * (1.0 - math.exp(-beta * depth))
    return penalty


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


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside.
    For a regular hexagon of side S, apothem = S*sqrt(3)/2.
    A point p is inside iff for each of the 6 outward normals n_k, (p-c)·n_k <= apothem.
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


def compute_min_outer_side_smooth(inner, outer_center, outer_angle, beta=30.0):
    """Smooth version of compute_min_outer_side for gradient-based optimization."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    all_projs = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                all_projs.append(proj)
    sm = smooth_max(all_projs, beta)
    return (2.0 / math.sqrt(3.0)) * sm


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
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


def pairwise_overlap_penalty(polys, soft_beta=50.0):
    """
    Compute a smooth penalty for pairwise overlaps using overlap depth.
    """
    return smooth_overlap_penalty(polys, beta=soft_beta)


def build_lattice_packing(scale, rot_in_deg, rot_out_deg, pattern='ring12'):
    """
    Build a hexagonal lattice packing of 12 unit hexagons.
    pattern: 'ring12' (center+6+5), 'hex6+6' (two rings of 6), 'rect3x4' (rectangular grid)
    """
    a = 2.0 * scale  # lattice constant
    
    positions = []
    
    if pattern == 'ring12':
        # Center + Ring 1 (6) + Ring 2 (5)
        positions.append((0.0, 0.0))
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            positions.append((x, y))
        for k in range(5):
            theta = math.radians(k * 60.0)
            x = 2.0 * a * math.cos(theta)
            y = 2.0 * a * math.sin(theta)
            positions.append((x, y))
    elif pattern == 'hex6+6':
        # Two full rings: Ring 1 (6) + Ring 2 (6) - no center
        # This gives a hexagonal ring of 12
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            positions.append((x, y))
        for k in range(6):
            theta = math.radians(k * 60.0 + 30.0)
            x = 2.0 * a * math.cos(theta)
            y = 2.0 * a * math.sin(theta)
            positions.append((x, y))
    elif pattern == 'rect3x4':
        # 3 rows x 4 columns rectangular arrangement
        # Row spacing = a * sin(60) = a * sqrt(3)/2
        # Column spacing = a
        row_spacing = a * math.sqrt(3) / 2.0
        for row in range(3):
            for col in range(4):
                x = col * a
                y = row * row_spacing
                if row % 2 == 1:
                    x += a / 2.0  # offset alternating rows
                positions.append((x - 1.5 * a, y - row_spacing))
    else:
        # Default to ring12
        positions.append((0.0, 0.0))
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            positions.append((x, y))
        for k in range(5):
            theta = math.radians(k * 60.0)
            x = 2.0 * a * math.cos(theta)
            y = 2.0 * a * math.sin(theta)
            positions.append((x, y))
    
    # Build inner list with rotation
    inner = []
    for (x, y) in positions:
        inner.append((x, y, rot_in_deg))
    
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out_deg)
    
    return inner, outer_center, outer_side, rot_out_deg


def params_to_inner(params):
    """Convert flat parameter vector to inner hexagon list."""
    inner = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        ang = params[i * 3 + 2]
        inner.append((x, y, ang))
    return inner


def objective(params, pattern='ring12'):
    """Objective function for optimization. Returns the outer side length,
    or a large penalty if the configuration is invalid."""
    if time.time() > DEADLINE:
        return 1e6
    
    scale, rot_in_deg, rot_out_deg = params
    
    # Constrain scale to reasonable range
    if scale < 0.7 or scale > 3.5:
        return 1e6
    
    inner, outer_center, outer_side, outer_angle = build_lattice_packing(scale, rot_in_deg, rot_out_deg, pattern)
    
    # Check validity
    if not check_validity(inner, outer_center, outer_side, outer_angle):
        return 1e6
    
    return outer_side


def free_placement_objective(params, outer_center=(0.0, 0.0), outer_angle=0.0,
                               overlap_weight=100.0):
    """
    Objective for free placement optimization using differential evolution.
    params: 36-dim vector [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    Returns: outer_side + overlap_penalty (smooth)
    """
    if time.time() > DEADLINE:
        return 1e6
    
    inner = params_to_inner(params)
    
    # Compute minimum outer side needed
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Compute overlap penalty (smooth)
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    overlap_pen = smooth_overlap_penalty(polys, beta=50.0)
    
    return outer_side + overlap_weight * overlap_pen


def optimize_topology():
    """Search over lattice parameters to find the best valid packing."""
    best_side = 1e6
    best_params = (1.0, 0.0, 0.0)
    best_pattern = 'ring12'
    
    patterns = ['ring12', 'hex6+6', 'rect3x4']
    
    # Coarse grid search over all patterns
    scale_vals = np.linspace(0.8, 2.5, 15)
    rot_in_vals = [0.0, 15.0, 30.0, 45.0]
    rot_out_vals = [0.0, 15.0, 30.0]
    
    for pattern in patterns:
        for scale in scale_vals:
            for rot_in in rot_in_vals:
                for rot_out in rot_out_vals:
                    if time.time() > DEADLINE:
                        return best_side, best_params, best_pattern
                    
                    side = objective([scale, rot_in, rot_out], pattern)
                    if side < best_side:
                        best_side = side
                        best_params = (scale, rot_in, rot_out)
                        best_pattern = pattern
    
    # Local refinement using scipy for the best pattern
    if best_side < 1e5:
        try:
            def obj_wrapped(params):
                return objective(params, best_pattern)
            
            result = minimize(obj_wrapped, best_params, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < best_side:
                best_side = result.fun
                best_params = tuple(result.x)
        except Exception:
            pass
    
    return best_side, best_params, best_pattern


def optimize_free_placement(max_time=100.0):
    """
    Optimize free placement of 12 hexagons using differential evolution.
    Returns (best_side, best_inner, best_outer_center, best_outer_angle)
    """
    t_start = time.time()
    best_side = 1e6
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Bounds for each parameter: x, y in [-5, 5], angle in [-60, 60]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((-60.0, 60.0))  # angle
    
    # Try a few outer angles with multiple seeds
    outer_angles = [0.0, 15.0, 30.0]
    seeds = [42, 123, 456]
    
    for outer_angle in outer_angles:
        if time.time() - t_start > max_time:
            break
        
        for seed in seeds:
            if time.time() - t_start > max_time:
                break
            
            try:
                result = differential_evolution(
                    lambda p, oa=outer_angle: free_placement_objective(p, (0.0, 0.0), oa),
                    bounds=bounds,
                    maxiter=80,
                    popsize=20,
                    tol=1e-5,
                    mutation=(0.4, 1.0),
                    recombination=0.9,
                    seed=seed,
                    polish=True,
                    polish_options={'method': 'Nelder-Mead', 'options': {'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7}},
                    init='latinhypercube'
                )
                
                if result.fun < 1e5:
                    inner = params_to_inner(result.x)
                    outer_side = compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
                    
                    if check_validity(inner, (0.0, 0.0), outer_side, outer_angle):
                        if outer_side < best_side:
                            best_side = outer_side
                            best_inner = inner
                            best_outer_center = (0.0, 0.0)
                            best_outer_angle = outer_angle
            except Exception:
                pass
    
    # Final polish: if we found a valid solution, do a tight local optimization
    if best_inner is not None and time.time() - t_start < max_time - 10:
        try:
            def tight_obj(params):
                inner = params_to_inner(params)
                outer_side = compute_min_outer_side(inner, (0.0, 0.0), best_outer_angle)
                polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
                overlap_pen = smooth_overlap_penalty(polys, beta=100.0)
                return outer_side + 200.0 * overlap_pen
            
            initial = np.array([p for hex_ in best_inner for p in hex_])
            result = minimize(
                tight_obj, initial, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if result.fun < 1e5:
                inner = params_to_inner(result.x)
                outer_side = compute_min_outer_side(inner, (0.0, 0.0), best_outer_angle)
                
                if check_validity(inner, (0.0, 0.0), outer_side, best_outer_angle):
                    if outer_side < best_side:
                        best_side = outer_side
                        best_inner = inner
        except Exception:
            pass
    
    return best_side, best_inner, best_outer_center, best_outer_angle


def construct():
    t0 = time.time()
    
    # First try lattice packing optimization (fast, ~5-10s)
    best_lattice_side = 1e6
    best_lattice_result = None
    try:
        best_side, best_params, best_pattern = optimize_topology()
        if best_side < 1e5:
            best_lattice_side = best_side
            scale, rot_in_deg, rot_out_deg = best_params
            inner, outer_center, outer_side, outer_angle = build_lattice_packing(
                scale, rot_in_deg, rot_out_deg, best_pattern)
            best_lattice_result = (inner, outer_center, outer_side, outer_angle)
    except Exception:
        pass
    
    # Then try free placement optimization with differential evolution
    best_free_side = 1e6
    best_free_result = None
    remaining_time = DEADLINE - time.time()
    if remaining_time > 30:  # Only if we have enough time
        try:
            free_max_time = min(remaining_time - 10, 120.0)
            best_side, best_inner, best_outer_center, best_outer_angle = optimize_free_placement(
                max_time=free_max_time)
            if best_side < 1e5 and best_inner is not None:
                best_free_side = best_side
                best_free_result = (best_inner, best_outer_center, best_side, best_outer_angle)
        except Exception:
            pass
    
    # Return the best result
    if best_free_side < best_lattice_side:
        if best_free_result is not None:
            return best_free_result
    elif best_lattice_result is not None:
        return best_lattice_result
    
    # Fallback: simple hexagonal cluster
    a = 2.0
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    
    inner = [(x, y, 0.0) for (x, y) in positions]
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, 0.0)
    return inner, outer_center, outer_side, 0.0
