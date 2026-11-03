"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses SLSQP with SAT-based constraints and multi-start optimization.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

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


def sat_overlap(A, B):
    """SAT: True if two convex polygons overlap (strictly, not just touching)."""
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


def min_distance_polygons(A, B):
    """Minimum distance between two convex polygons (0 if overlapping)."""
    if sat_overlap(A, B):
        return 0.0
    min_d = float('inf')
    for i in range(6):
        x1, y1 = A[i]
        x2, y2 = A[(i+1) % 6]
        for j in range(6):
            x3, y3 = B[j]
            x4, y4 = B[(j+1) % 6]
            # Point-to-segment distances
            for (px, py, sx1, sy1, sx2, sy2) in [(x1, y1, x3, y3, x4, y4), (x3, y3, x1, y1, x2, y2)]:
                dx = sx2 - sx1
                dy = sy2 - sy1
                len_sq = dx*dx + dy*dy
                if len_sq < 1e-12:
                    d = math.sqrt((px-sx1)**2 + (py-sy1)**2)
                else:
                    t = max(0, min(1, ((px-sx1)*dx + (py-sy1)*dy) / len_sq))
                    projx = sx1 + t*dx
                    projy = sy1 + t*dy
                    d = math.sqrt((px-projx)**2 + (py-projy)**2)
                min_d = min(min_d, d)
    return min_d


def compute_min_outer_side(centers, angles, ocx, ocy, otheta_deg):
    """Compute minimum outer hexagon side for given configuration."""
    otheta = math.radians(otheta_deg)
    max_proj = 0.0
    for k in range(6):
        normal_angle = otheta + math.pi / 6 + k * math.pi / 3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle = math.radians(angles[i])
            for vi in range(6):
                vert_angle = angle + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / math.sqrt(3)


def check_validity(centers, angles, ocx, ocy, outer_side, outer_angle):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(ocx, ocy, outer_side, outer_angle)
    polys = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        p = hex_vertices(cx, cy, 1.0, ang)
        polys.append(p)
    # Check all vertices of inner hexagons are inside outer
    for p in polys:
        for (px, py) in p:
            inside = True
            for i in range(6):
                x1, y1 = outer[i]
                x2, y2 = outer[(i+1) % 6]
                cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
                if cross < -1e-9:
                    inside = False
                    break
            if not inside:
                return False
    # Check pairwise non-overlap
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if sat_overlap(polys[i], polys[j]):
                return False
    return True


def params_to_inner(x):
    """Convert parameter vector to centers, angles, outer params."""
    n = N
    centers = [[x[3*i], x[3*i+1]] for i in range(n)]
    angles = [x[3*i+2] for i in range(n)]
    ocx, ocy, otheta = x[3*n], x[3*n+1], x[3*n+2]
    return centers, angles, ocx, ocy, otheta


def objective(x):
    """Objective: minimize outer hexagon side length."""
    if time.time() > DEADLINE - 10:
        return 1e10
    centers, angles, ocx, ocy, otheta = params_to_inner(x)
    return compute_min_outer_side(centers, angles, ocx, ocy, otheta)


def constraints(x):
    """Constraints: pairwise non-overlap (min distance > 0) and containment."""
    centers, angles, ocx, ocy, otheta = params_to_inner(x)
    outer = hex_vertices(ocx, ocy, compute_min_outer_side(centers, angles, ocx, ocy, otheta), otheta)
    cons = []
    
    # Pairwise non-overlap: min distance > 0.01
    for i in range(N):
        for j in range(i+1, N):
            pi = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            pj = hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
            dist = min_distance_polygons(pi, pj)
            cons.append(dist - 0.01)
    
    # Containment: each inner vertex must be inside outer (use distance to boundary)
    for i in range(N):
        p = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        for (px, py) in p:
            # Check if point is inside outer hexagon
            inside = True
            for k in range(6):
                x1, y1 = outer[k]
                x2, y2 = outer[(k+1) % 6]
                cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                cons.append(-1.0)  # Violation
            else:
                cons.append(0.1)  # Satisfied
    
    return cons


def build_lattice_packing(pattern='ring12', scale=1.001, rot_in=0.0, rot_out=0.0):
    """Build a hexagonal lattice packing of 12 unit hexagons."""
    a = 2.0 * scale
    positions = []
    
    if pattern == 'ring12':
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
        row_spacing = a * math.sqrt(3) / 2.0
        for row in range(3):
            for col in range(4):
                x = col * a
                y = row * row_spacing
                if row % 2 == 1:
                    x += a / 2.0
                positions.append((x - 1.5 * a, y - row_spacing))
    else:
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
    
    centers = [list(p) for p in positions]
    angles = [rot_in] * 12
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(centers, angles, 0.0, 0.0, rot_out)
    
    return centers, angles, outer_center, outer_side, rot_out


def optimize_slsqp(centers, angles, max_iters=100):
    """Optimize using SLSQP with constraints."""
    n = N
    x0 = []
    for i in range(n):
        x0.extend([centers[i][0], centers[i][1], angles[i]])
    x0.extend([0.0, 0.0, 0.0])
    
    bounds = []
    for i in range(n):
        bounds.extend([(-6, 6), (-6, 6), (-45, 45)])
    bounds.extend([(-3, 3), (-3, 3), (-45, 45)])
    
    try:
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds,
                         constraints=[{'type': 'ineq', 'fun': constraints}],
                         options={'maxiter': max_iters, 'ftol': 1e-10})
        
        if result.fun < 1e9:
            c, a, ocx, ocy, otheta = params_to_inner(result.x)
            side = compute_min_outer_side(c, a, ocx, ocy, otheta)
            return c, a, ocx, ocy, otheta, side
    except Exception:
        pass
    
    return None


def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
    """High-precision polish of outer hexagon parameters."""
    best_side = compute_min_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
    best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta
    
    scales = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for scale in scales:
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    for da_deg in [-scale*10, 0, scale*10]:
                        otheta = best_otheta + da_deg
                        side = compute_min_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, otheta)
                        if side < best_side - 1e-14:
                            best_side = side
                            best_ocx += dx
                            best_ocy += dy
                            best_otheta = otheta
                            improved = True
    
    return best_ocx, best_ocy, best_otheta, best_side


def generate_random_configurations(num_configs=5, seed=42):
    """Generate random initial configurations."""
    rng = random.Random(seed)
    configs = []
    for _ in range(num_configs):
        centers = []
        for i in range(N):
            r = rng.uniform(0, 4)
            theta = rng.uniform(0, 2 * math.pi)
            cx = r * math.cos(theta)
            cy = r * math.sin(theta)
            centers.append([cx, cy])
        angles = [rng.uniform(-45, 45) for _ in range(N)]
        configs.append((centers, angles))
    return configs


def construct():
    t0 = time.time()
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple lattice-based configurations with SLSQP optimization
    lattice_patterns = ['ring12', 'hex6+6', 'rect3x4']
    
    for pattern in lattice_patterns:
        if time.time() > DEADLINE - 30:
            break
        
        for scale in [1.001, 1.002, 1.003, 1.005]:
            if time.time() > DEADLINE - 30:
                break
            
            for rot_in in [0.0, 30.0, 15.0]:
                if time.time() > DEADLINE - 30:
                    break
                
                centers, angles, outer_center, outer_side, rot_out = build_lattice_packing(
                    pattern, scale, rot_in, 0.0)
                
                # Try SLSQP optimization with constraints
                opt_result = optimize_slsqp(centers, angles, max_iters=80)
                
                if opt_result is not None:
                    c, a, ocx, ocy, otheta, side = opt_result
                    
                    # Polish outer hexagon
                    ocx, ocy, otheta, side = polish_outer(c, a, ocx, ocy, otheta)
                    
                    # Verify validity
                    if check_validity(c, a, ocx, ocy, side, otheta):
                        if side < best_side:
                            best_side = side
                            best_result = (c, a, ocx, ocy, side, otheta)
    
    # Try random configurations
    random_configs = generate_random_configurations(num_configs=6, seed=42)
    for centers, angles in random_configs:
        if time.time() > DEADLINE - 15:
            break
        
        opt_result = optimize_slsqp(centers, angles, max_iters=60)
        
        if opt_result is not None:
            c, a, ocx, ocy, otheta, side = opt_result
            
            ocx, ocy, otheta, side = polish_outer(c, a, ocx, ocy, otheta)
            
            if check_validity(c, a, ocx, ocy, side, otheta):
                if side < best_side:
                    best_side = side
                    best_result = (c, a, ocx, ocy, side, otheta)
    
    # If we found a good result, return it
    if best_result is not None:
        c, a, ocx, ocy, side, otheta = best_result
        inner = [(c[i][0], c[i][1], a[i]) for i in range(N)]
        outer_center = (ocx, ocy)
        return inner, outer_center, side, otheta
    
    # Fallback: simple hexagonal cluster with slight expansion to ensure disjointness
    a = 2.002
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    
    centers = [list(p) for p in positions]
    angles = [0.0] * 12
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(centers, angles, 0.0, 0.0, 0.0)
    
    inner = [(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
    return inner, outer_center, outer_side, 0.0
