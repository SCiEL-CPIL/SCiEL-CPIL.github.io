"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Combines topology-aware search (from current) with hard-constraint SLSQP
optimization and smooth max objective (from inspiration).
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill
random.seed(42)
np.random.seed(42)

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
    """True if (px,py) is inside or on boundary of convex polygon."""
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
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
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

def compute_min_outer_side_smooth(inner, outer_center, outer_angle, beta=100.0):
    """Smooth approximation of min outer side using log-sum-exp."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    projs = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                projs.append(proj)
    
    if not projs:
        return 0.0
    
    max_p = max(projs)
    exps = [math.exp(beta * (p - max_p)) for p in projs]
    log_sum = math.log(sum(exps))
    smooth_max = max_p + log_sum / beta
    
    return (2.0 / math.sqrt(3.0)) * max(0.0, smooth_max)

def sat_gap(A, B):
    """Compute maximum SAT gap between two polygons. Positive = separated, negative = overlap."""
    max_gap = -float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9:
                continue
            nx, ny = nx/norm, ny/norm
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB:
                gap = minB - maxA
            elif maxB < minA:
                gap = minA - maxB
            else:
                gap = -(min(maxA, maxB) - max(minA, minB))
            if gap > max_gap:
                max_gap = gap
    return max_gap

def compute_overlap_penalty(inner, outer_center, outer_side, outer_angle):
    """Compute smooth penalty for overlaps and boundary violations."""
    penalty = 0.0
    polys = [hex_vertices(x, y, 1.0, ang) for x, y, ang in inner]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0:
                penalty += (-gap) ** 2
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                penalty += 0.1
    return penalty

# --- Topology builders ---

def build_ring12(scale, inner_rot, outer_rot):
    inner = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append((x, y, inner_rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot):
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
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex_lattice(r1, r2, r3, outer_rot):
    """Three concentric rings: 1 center + 6 inner + 5 outer = 12."""
    inner = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner.append((x, y, 0.0))
    for k in range(5):
        theta = math.radians(k * 72.0 + 18.0)
        x = r3 * math.cos(theta)
        y = r3 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner, (0.0, 0.0), outer_rot

def build_linear_chain(spacing, outer_rot):
    inner = []
    for k in range(12):
        x = (k - 5.5) * spacing
        inner.append((x, 0.0, 0.0))
    return inner, (0.0, 0.0), outer_rot

def build_hexagonal_cluster(r1, r2, outer_rot):
    """1 center + 6 at r1 + 5 at r2."""
    inner = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner.append((x, y, 0.0))
    for k in range(5):
        theta = math.radians(k * 72.0 + 18.0)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner, (0.0, 0.0), outer_rot

# --- Optimization ---

def optimize_topology(topology, x0, bounds, outer_rot, time_budget):
    """Optimize a single topology using grid search + Nelder-Mead."""
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
            r1, r2, r3 = params
            if r1 < 1.0 or r2 < r1 + 0.5 or r3 < r2 + 0.5:
                return 1000.0
            inner, center, orot = build_hex_lattice(r1, r2, r3, outer_rot)
        elif topology == 'linear':
            spacing = params[0]
            if spacing < 1.5:
                return 1000.0
            inner, center, orot = build_linear_chain(spacing, outer_rot)
        elif topology == 'hex_cluster':
            r1, r2 = params
            if r1 < 1.0 or r2 < r1 + 0.5:
                return 1000.0
            inner, center, orot = build_hexagonal_cluster(r1, r2, outer_rot)
        else:
            return 1000.0
        
        side = compute_min_outer_side(inner, center, orot)
        if not check_validity(inner, center, side, orot):
            return side + 10.0
        return side
    
    best_x = list(x0)
    best_f = objective(best_x)
    
    # Coarse grid search
    n_grid = 5
    dims = len(x0)
    if dims == 2:
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
    elif dims == 3:
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
    elif dims == 4:
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
    elif dims == 1:
        grid1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        for g1 in grid1:
            f = objective([g1])
            if f < best_f:
                best_f = f
                best_x = [g1]
            if time.time() - start_time > time_budget:
                return best_x, best_f
    
    # Nelder-Mead refinement
    if time.time() - start_time < time_budget:
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

def polish_configuration(inner, center, outer_angle, time_budget=10.0):
    """High-precision polish using SLSQP on all parameters with smooth objective."""
    start_time = time.time()
    
    n_params = len(inner) * 3 + 3
    
    def params_to_inner(params):
        inner_out = []
        for i in range(len(inner)):
            x = params[i*3]
            y = params[i*3+1]
            ang = params[i*3+2]
            inner_out.append((x, y, ang))
        outer_cx = params[-3]
        outer_cy = params[-2]
        outer_ang = params[-1]
        return inner_out, (outer_cx, outer_cy), outer_ang
    
    def objective(params):
        if time.time() - start_time > time_budget:
            return float('inf')
        
        inner_out, outer_c, outer_a = params_to_inner(params)
        
        side = compute_min_outer_side_smooth(inner_out, outer_c, outer_a, beta=100.0)
        
        if not check_validity(inner_out, outer_c, side, outer_a):
            penalty = compute_overlap_penalty(inner_out, outer_c, side, outer_a)
            return side + 5.0 * penalty
        
        return side
    
    x0 = []
    for x, y, ang in inner:
        x0.extend([x, y, ang])
    x0.extend([center[0], center[1], outer_angle])
    
    bounds = []
    for _ in range(len(inner)):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((-180, 180))
    bounds.append((-10, 10))
    bounds.append((-10, 10))
    bounds.append((-180, 180))
    
    init_val = objective(x0)
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 600, 'ftol': 1e-10, 'maxls': 50}
        )
        if result.fun < init_val:
            inner_out, outer_c, outer_a = params_to_inner(result.x)
            exact_side = compute_min_outer_side(inner_out, outer_c, outer_a)
            if check_validity(inner_out, outer_c, exact_side, outer_a):
                return inner_out, outer_c, outer_a, exact_side
    except Exception:
        pass
    
    return inner, center, outer_angle, init_val

def final_polish(inner, center, outer_angle, time_budget=8.0):
    """Final high-precision Nelder-Mead polish."""
    start_time = time.time()
    
    def params_to_inner(params):
        inner_out = []
        for i in range(len(inner)):
            x = params[i*3]
            y = params[i*3+1]
            ang = params[i*3+2]
            inner_out.append((x, y, ang))
        outer_cx = params[-3]
        outer_cy = params[-2]
        outer_ang = params[-1]
        return inner_out, (outer_cx, outer_cy), outer_ang
    
    def objective(params):
        if time.time() - start_time > time_budget:
            return float('inf')
        inner_out, outer_c, outer_a = params_to_inner(params)
        side = compute_min_outer_side(inner_out, outer_c, outer_a)
        if not check_validity(inner_out, outer_c, side, outer_a):
            return side + 10.0
        return side
    
    x0 = []
    for x, y, ang in inner:
        x0.extend([x, y, ang])
    x0.extend([center[0], center[1], outer_angle])
    
    init_val = objective(x0)
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'maxfev': 8000,
                'xatol': 1e-9,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        if result.fun < init_val:
            inner_out, outer_c, outer_a = params_to_inner(result.x)
            exact_side = compute_min_outer_side(inner_out, outer_c, outer_a)
            if check_validity(inner_out, outer_c, exact_side, outer_a):
                return inner_out, outer_c, outer_a, exact_side
    except Exception:
        pass
    
    return inner, center, outer_angle, init_val

def construct():
    start_time = time.time()
    
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_side = float('inf')
    
    # Phase 1: Try predefined topologies with optimization
    topologies = [
        ('ring12', [2.0, 0.0], [(0.8, 2.5), (0, 360)]),
        ('hex6+6', [1.5, 3.0, 0.0, 30.0], [(0.8, 2.5), (1.5, 4.0), (0, 360), (0, 360)]),
        ('rect3x4', [2.0, 2.0, 0.0], [(1.5, 3.0), (1.5, 3.0), (0, 360)]),
        ('hex_lattice', [2.0, 3.5, 5.0], [(1.0, 3.0), (2.5, 5.0), (4.0, 7.0)]),
        ('linear', [2.0], [(1.5, 3.0)]),
        ('hex_cluster', [2.0, 4.0], [(1.0, 3.0), (3.0, 6.0)]),
    ]
    
    for topology, x0, bounds in topologies:
        if time.time() > DEADLINE - 35:
            break
        
        remaining = DEADLINE - time.time()
        time_budget = min(12.0, remaining * 0.25)
        
        x_opt, f_opt = optimize_topology(topology, x0, bounds, 0.0, time_budget)
        
        if f_opt < best_side:
            best_side = f_opt
            if topology == 'ring12':
                scale, inner_rot = x_opt
                inner, center, outer_angle = build_ring12(scale, inner_rot, 0.0)
            elif topology == 'hex6+6':
                scale1, scale2, rot1, rot2 = x_opt
                inner, center, outer_angle = build_hex6_plus6(scale1, scale2, rot1, rot2, 0.0)
            elif topology == 'rect3x4':
                sx, sy, rot = x_opt
                inner, center, outer_angle = build_rect3x4(sx, sy, rot, 0.0)
            elif topology == 'hex_lattice':
                r1, r2, r3 = x_opt
                inner, center, outer_angle = build_hex_lattice(r1, r2, r3, 0.0)
            elif topology == 'linear':
                spacing = x_opt[0]
                inner, center, outer_angle = build_linear_chain(spacing, 0.0)
            elif topology == 'hex_cluster':
                r1, r2 = x_opt
                inner, center, outer_angle = build_hexagonal_cluster(r1, r2, 0.0)
            else:
                continue
            
            if check_validity(inner, center, best_side, outer_angle):
                best_inner = inner
                best_center = center
                best_outer_angle = outer_angle
    
    # Phase 2: Differential evolution for global search
    if time.time() < DEADLINE - 25:
        def de_objective(params):
            """DE objective: 12 hexagons, each with x,y,angle + outer center x,y + outer angle = 39 params."""
            inner_out = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                ang = params[i*3+2]
                inner_out.append((x, y, ang))
            outer_cx = params[-3]
            outer_cy = params[-2]
            outer_ang = params[-1]
            
            side = compute_min_outer_side(inner_out, (outer_cx, outer_cy), outer_ang)
            if not check_validity(inner_out, (outer_cx, outer_cy), side, outer_ang):
                return side + 10.0
            return side
        
        # Initial bounds
        de_bounds = []
        for _ in range(N):
            de_bounds.append((-8, 8))
            de_bounds.append((-8, 8))
            de_bounds.append((-90, 90))
        de_bounds.append((-8, 8))
        de_bounds.append((-8, 8))
        de_bounds.append((-90, 90))
        
        # Use best_inner as seed for DE
        if best_inner is not None:
            x0_seed = []
            for x, y, ang in best_inner:
                x0_seed.extend([x, y, ang])
            x0_seed.extend([best_center[0], best_center[1], best_outer_angle])
        else:
            x0_seed = [random.uniform(-5, 5) for _ in range(39)]
        
        de_time = min(20.0, DEADLINE - time.time() - 15)
        if de_time > 5:
            try:
                de_result = differential_evolution(
                    de_objective,
                    de_bounds,
                    seed=42,
                    maxiter=50,
                    popsize=10,
                    tol=1e-6,
                    polish=False,
                    update_interval=5
                )
                if de_result.fun < best_side:
                    params = de_result.x
                    inner_out = []
                    for i in range(N):
                        inner_out.append((params[i*3], params[i*3+1], params[i*3+2]))
                    outer_cx, outer_cy, outer_ang = params[-3], params[-2], params[-1]
                    exact_side = compute_min_outer_side(inner_out, (outer_cx, outer_cy), outer_ang)
                    if check_validity(inner_out, (outer_cx, outer_cy), exact_side, outer_ang):
                        best_inner = inner_out
                        best_center = (outer_cx, outer_cy)
                        best_outer_angle = outer_ang
                        best_side = exact_side
            except Exception:
                pass
    
    # Phase 3: Polish the best configuration
    if best_inner is not None and time.time() < DEADLINE - 15:
        polish_time = min(12.0, DEADLINE - time.time() - 8)
        if polish_time > 3:
            polished_inner, polished_center, polished_angle, polished_side = polish_configuration(
                best_inner, best_center, best_outer_angle, polish_time
            )
            if polished_side < best_side:
                if check_validity(polished_inner, polished_center, polished_side, polished_angle):
                    best_inner = polished_inner
                    best_center = polished_center
                    best_outer_angle = polished_angle
                    best_side = polished_side
    
    # Phase 4: Final high-precision Nelder-Mead polish
    if best_inner is not None and time.time() < DEADLINE - 5:
        fm_time = min(8.0, DEADLINE - time.time() - 2)
        if fm_time > 2:
            fm_inner, fm_center, fm_angle, fm_side = final_polish(
                best_inner, best_center, best_outer_angle, fm_time
            )
            if fm_side < best_side:
                if check_validity(fm_inner, fm_center, fm_side, fm_angle):
                    best_inner = fm_inner
                    best_center = fm_center
                    best_outer_angle = fm_angle
                    best_side = fm_side
    
    # Final fallback if nothing worked
    if best_inner is None:
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
        best_inner = [(px - cx, py - cy, angle) for px, py, angle in positions]
        best_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_min_outer_side(best_inner, best_center, best_outer_angle)
    
    # Compute final side and verify
    final_side = compute_min_outer_side(best_inner, best_center, best_outer_angle)
    
    if not check_validity(best_inner, best_center, final_side, best_outer_angle):
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
        best_inner = [(px - cx, py - cy, angle) for px, py, angle in positions]
        best_center = (0.0, 0.0)
        best_outer_angle = 0.0
        final_side = compute_min_outer_side(best_inner, best_center, best_outer_angle)
    
    inner_list = []
    for x, y, angle in best_inner:
        inner_list.append([float(x), float(y), float(angle)])
    
    return inner_list, (float(best_center[0]), float(best_center[1])), float(final_side), float(best_outer_angle)
