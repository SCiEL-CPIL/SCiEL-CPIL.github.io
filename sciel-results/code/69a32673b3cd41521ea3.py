"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses multi-start SLSQP with smooth max objective and contact-graph guided initializations.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 130.0  # safety margin before 150s kill


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


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    # The apothem (inradius) of the outer hexagon with side s is s*sqrt(3)/2
    # The support function: max over all inner vertices of projection onto 6 outward normals
    normals = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
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
    
    # The apothem of outer hexagon = s * sqrt(3)/2, so s = 2*apothem/sqrt(3)
    return (2.0 / math.sqrt(3.0)) * max_proj


def compute_min_outer_side_smooth(inner, outer_center, outer_angle, beta=20.0):
    """Smooth version using log-sum-exp for differentiable optimization."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    all_projs = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                all_projs.append(proj)
    
    if not all_projs:
        return 0.0
    
    max_p = max(all_projs)
    # Log-sum-exp for smooth max
    exps = [math.exp(beta * (p - max_p)) for p in all_projs]
    s = sum(exps)
    if s < 1e-300:
        return (2.0 / math.sqrt(3.0)) * max_p
    smooth_max = max_p + math.log(s) / beta
    return (2.0 / math.sqrt(3.0)) * smooth_max


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


def build_hex_cluster():
    """Center + ring of 6 + partial ring of 5."""
    a = 2.0
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0 + 30.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    return [(x, y, 0.0) for (x, y) in positions]


def build_two_rings():
    """Two rings of 6 hexagons."""
    a = 2.0
    positions = []
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(6):
        theta = math.radians(k * 60.0 + 30.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    return [(x, y, 0.0) for (x, y) in positions]


def build_grid_3x4():
    """3x4 grid with hexagonal offset."""
    a = 2.0
    row_spacing = a * math.sqrt(3) / 2.0
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * a
            y = row * row_spacing
            if row % 2 == 1:
                x += a / 2.0
            positions.append((x - 1.5 * a, y - row_spacing))
    return [(x, y, 0.0) for (x, y) in positions]


def build_linear():
    """Two rows of 6."""
    a = 2.0
    positions = []
    for col in range(6):
        positions.append((col * a - 2.5 * a, 0.0))
        positions.append((col * a - 2.5 * a + a/2, a * math.sqrt(3)/2))
    return [(x, y, 0.0) for (x, y) in positions]


def build_compact():
    """Compact arrangement: 2-3-4-3 pattern."""
    a = 2.0
    h = a * math.sqrt(3) / 2
    positions = []
    # Row 0: 2 hexagons
    positions.append((-a/2, 0.0))
    positions.append((a/2, 0.0))
    # Row 1: 3 hexagons
    positions.append((-a, h))
    positions.append((0.0, h))
    positions.append((a, h))
    # Row 2: 4 hexagons
    positions.append((-1.5*a, 2*h))
    positions.append((-0.5*a, 2*h))
    positions.append((0.5*a, 2*h))
    positions.append((1.5*a, 2*h))
    # Row 3: 3 hexagons
    positions.append((-a, 3*h))
    positions.append((0.0, 3*h))
    positions.append((a, 3*h))
    return [(x, y, 0.0) for (x, y) in positions]


def objective(params, outer_center=(0.0, 0.0), outer_angle=0.0, beta=20.0):
    """
    Objective: minimize outer side length with smooth max.
    params: 36-dim vector [x0,y0,a0, x1,y1,a1, ...]
    """
    inner = []
    for i in range(N):
        inner.append((params[i*3], params[i*3+1], params[i*3+2]))
    
    side = compute_min_outer_side_smooth(inner, outer_center, outer_angle, beta)
    
    # Add penalty for overlaps using smooth approximation
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap_sat(polys[i], polys[j]):
                # Compute approximate overlap depth via center distance
                xi, yi = params[i*3], params[i*3+1]
                xj, yj = params[j*3], params[j*3+1]
                dx, dy = xj-xi, yj-yi
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum center distance for non-overlap is 2*sqrt(3)/2 = sqrt(3) for aligned hexagons
                # But with rotation it's complex; use a simple penalty
                overlap_pen += max(0, 2.0 - dist) ** 2
    
    # Boundary penalty: push inward if vertices are outside
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    outer_poly = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    boundary_pen = 0.0
    for i in range(N):
        p = polys[i]
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer_poly):
                cx, cy = outer_center
                dx, dy = px-cx, py-cy
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 1e-9:
                    boundary_pen += 0.1 * (dist - outer_side * math.sqrt(3)/2) ** 2
    
    return side + 5.0 * overlap_pen + 5.0 * boundary_pen


def optimize_slsqp(inner_init, outer_center=(0.0, 0.0), outer_angle=0.0, 
                   max_iter=200, beta=20.0):
    """SLSQP optimization for a given initial configuration."""
    params = np.array([v for item in inner_init for v in item])
    
    def obj(p):
        if time.time() > DEADLINE:
            return 1e6
        return objective(p, outer_center, outer_angle, beta)
    
    try:
        result = minimize(obj, params, method='SLSQP',
                         options={'maxiter': max_iter, 'ftol': 1e-8})
        if result.fun < 1e5:
            inner = []
            for i in range(N):
                inner.append((result.x[i*3], result.x[i*3+1], result.x[i*3+2]))
            outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
            if check_validity(inner, outer_center, outer_side, outer_angle):
                return outer_side, inner, outer_center, outer_angle
    except Exception:
        pass
    return None


def optimize_nelder_mead(inner_init, outer_center=(0.0, 0.0), outer_angle=0.0,
                         max_iter=300, beta=10.0):
    """Nelder-Mead optimization."""
    params = np.array([v for item in inner_init for v in item])
    
    def obj(p):
        if time.time() > DEADLINE:
            return 1e6
        return objective(p, outer_center, outer_angle, beta)
    
    try:
        result = minimize(obj, params, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < 1e5:
            inner = []
            for i in range(N):
                inner.append((result.x[i*3], result.x[i*3+1], result.x[i*3+2]))
            outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
            if check_validity(inner, outer_center, outer_side, outer_angle):
                return outer_side, inner, outer_center, outer_angle
    except Exception:
        pass
    return None


def polish_outer(inner, outer_center, outer_angle, n_iter=50):
    """
    Polish the outer hexagon: adjust center and angle to minimize side length.
    """
    cx, cy = outer_center
    best_side = compute_min_outer_side(inner, outer_center, outer_angle)
    best_cx, best_cy, best_angle = cx, cy, outer_angle
    
    for it in range(n_iter):
        if time.time() > DEADLINE:
            break
        improved = False
        for dcx in [-0.05, 0.05]:
            for dcy in [-0.05, 0.05]:
                for da in [-2.0, 0.0, 2.0]:
                    nc = (best_cx + dcx, best_cy + dcy)
                    na = best_angle + da
                    s = compute_min_outer_side(inner, nc, na)
                    if s < best_side - 1e-10:
                        best_side = s
                        best_cx, best_cy, best_angle = nc[0], nc[1], na
                        improved = True
        if not improved:
            break
    
    # Verify validity with polished outer
    if check_validity(inner, (best_cx, best_cy), best_side, best_angle):
        return best_side, inner, (best_cx, best_cy), best_angle
    else:
        # Fallback: recompute with original center/angle
        side = compute_min_outer_side(inner, outer_center, outer_angle)
        if check_validity(inner, outer_center, side, outer_angle):
            return side, inner, outer_center, outer_angle
        return None


def construct():
    t0 = time.time()
    
    best_side = 1e6
    best_result = None
    
    # Generate diverse initial configurations
    initial_configs = [
        build_hex_cluster(),
        build_two_rings(),
        build_grid_3x4(),
        build_linear(),
        build_compact(),
    ]
    
    rng = np.random.RandomState(42)
    
    # Add randomized perturbations
    for base in initial_configs[:3]:
        for r in range(2):
            noise = 0.3 + 0.2 * r
            perturbed = []
            for (x, y, ang) in base:
                perturbed.append((x + rng.uniform(-noise, noise),
                                 y + rng.uniform(-noise, noise),
                                 ang + rng.uniform(-15, 15)))
            initial_configs.append(perturbed)
    
    outer_angles = [0.0, 15.0, 30.0]
    
    # Phase 1: SLSQP from multiple starts
    for config in initial_configs:
        if time.time() > DEADLINE - 30:
            break
        for outer_angle in outer_angles:
            if time.time() > DEADLINE - 30:
                break
            try:
                result = optimize_slsqp(config, (0.0, 0.0), outer_angle,
                                        max_iter=150, beta=20.0)
                if result is not None:
                    side, inner, center, angle = result
                    if side < best_side:
                        best_side = side
                        best_result = (inner, center, side, angle)
            except Exception:
                pass
    
    # Phase 2: Nelder-Mead refinement from best found and other configs
    if best_result is not None:
        remaining = DEADLINE - time.time()
        if remaining > 20:
            try:
                inner_best = best_result[0]
                result = optimize_nelder_mead(inner_best, best_result[1], best_result[3],
                                              max_iter=200, beta=10.0)
                if result is not None:
                    side, inner, center, angle = result
                    if side < best_side:
                        best_side = side
                        best_result = (inner, center, side, angle)
            except Exception:
                pass
    
    # Phase 3: Polish outer hexagon
    if best_result is not None:
        remaining = DEADLINE - time.time()
        if remaining > 10:
            try:
                polished = polish_outer(best_result[0], best_result[1], best_result[3], n_iter=80)
                if polished is not None:
                    side, inner, center, angle = polished
                    if side < best_side:
                        best_side = side
                        best_result = (inner, center, side, angle)
            except Exception:
                pass
    
    # Fallback: simple hexagonal cluster
    if best_result is None:
        inner = build_hex_cluster()
        outer_center = (0.0, 0.0)
        outer_side = compute_min_outer_side(inner, outer_center, 0.0)
        return inner, outer_center, outer_side, 0.0
    
    return best_result
