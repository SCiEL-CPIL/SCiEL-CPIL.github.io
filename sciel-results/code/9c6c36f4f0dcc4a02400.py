"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses free-position optimization with penalty method and multi-restart.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 138.0  # safety margin before 150s kill

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

def polygons_overlap_sat(A, B, eps=1e-9):
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
            if maxA < minB - eps or maxB < minA - eps:
                return False
    return True

def compute_min_outer_side(inner_positions, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside.
    inner_positions: list of (x, y, angle_deg) with unit side."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner_positions:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj

def check_validity(inner_positions, outer_center, outer_side, outer_angle):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner_positions:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def objective(params, outer_center, outer_angle):
    """Penalty-based objective: minimize outer side + penalties for overlap and outside."""
    n_hex = len(params) // 3
    # Extract positions and angles
    centers = [(params[i*3], params[i*3+1]) for i in range(n_hex)]
    angles = [params[i*3+2] * 180.0 / math.pi for i in range(n_hex)]
    inner_positions = [(centers[i][0], centers[i][1], angles[i]) for i in range(n_hex)]
    
    # Compute required outer side
    side = compute_min_outer_side(inner_positions, outer_center, outer_angle)
    
    # Penalties
    penalty = 0.0
    outer = hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    polys = []
    for (x, y, ang) in inner_positions:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    
    # Check inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                penalty += 10.0
    
    # Check pairwise overlap
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                penalty += 10.0
    
    return side + penalty

def generate_initial_config(seed):
    """Generate an initial configuration using a honeycomb-like pattern with randomness."""
    rng = np.random.RandomState(seed)
    # Start with a structured 3x4 grid in hexagonal coordinates
    # Hex grid positions for 12 hexagons
    positions = []
    # Row 0: 4 hexagons
    for k in range(4):
        positions.append((k * 2.0, 0.0, 0.0))
    # Row 1: 4 hexagons (offset)
    for k in range(4):
        positions.append((k * 2.0 + 1.0, math.sqrt(3), 0.0))
    # Row 2: 4 hexagons
    for k in range(4):
        positions.append((k * 2.0, 2 * math.sqrt(3), 0.0))
    
    # Add small random perturbations
    params = []
    for (x, y, ang) in positions:
        x += rng.uniform(-0.3, 0.3)
        y += rng.uniform(-0.3, 0.3)
        ang += rng.uniform(-math.pi/12, math.pi/12)
        params.extend([x, y, ang])
    
    return params

def optimize_single(start_params, outer_center, outer_angle, max_iter=200):
    """Run L-BFGS-B optimization from a single starting point."""
    if time.time() > DEADLINE:
        return None
    
    result = minimize(
        objective,
        start_params,
        args=(outer_center, outer_angle),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-4}
    )
    
    if result.success or result.fun < 10.0:  # Accept if penalty is small
        return result.x, result.fun
    return None

def construct():
    t0 = time.time()
    best_side = 40.0
    best_params = None
    best_valid = False
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Try multiple random restarts
    for seed in range(50):
        if time.time() > DEADLINE:
            break
        
        start_params = generate_initial_config(seed)
        result = optimize_single(start_params, outer_center, outer_angle, max_iter=100)
        
        if result is not None:
            opt_params, opt_fun = result
            n_hex = len(opt_params) // 3
            inner_positions = []
            for i in range(n_hex):
                x = opt_params[i*3]
                y = opt_params[i*3+1]
                ang_deg = opt_params[i*3+2] * 180.0 / math.pi
                inner_positions.append((x, y, ang_deg))
            
            side = compute_min_outer_side(inner_positions, outer_center, outer_angle)
            
            # Check validity
            if check_validity(inner_positions, outer_center, side, outer_angle):
                if side < best_side:
                    best_side = side
                    best_params = inner_positions
                    best_valid = True
    
    if best_valid and best_params is not None:
        return best_params, outer_center, best_side, outer_angle
    
    # Fallback: linear arrangement
    inner = [(4.0 * k, 0.0, 0.0) for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
