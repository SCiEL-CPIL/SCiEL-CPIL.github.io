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

def polygon_overlap(A, B, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    for poly in [A, B]:
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
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
            
            if maxA < minB - tol or maxB < minA - tol:
                return False
    return True

def point_in_convex_polygon(px, py, poly, tol=1e-9):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < tol:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def validate_config(inner_params, outer_center, side, angle):
    """Validate that all inner hexagons are inside outer and don't overlap."""
    outer_poly = hex_vertices(outer_center[0], outer_center[1], side, angle)
    inner_polys = []
    
    for i in range(N):
        x = inner_params[i*3]
        y = inner_params[i*3 + 1]
        th = inner_params[i*3 + 2]
        poly = hex_vertices(x, y, 1.0, th)
        inner_polys.append(poly)
        
        # Check containment
        for px, py in poly:
            if not point_in_convex_polygon(px, py, outer_poly):
                return False
    
    # Check no overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if polygon_overlap(inner_polys[i], inner_polys[j]):
                return False
    
    return True

def generate_hex_lattice_configs():
    """Generate multiple hexagonal lattice topologies for 12 hexagons."""
    configs = []
    
    # Config 1: 3-4-3-2 pattern (hexagonal cluster)
    # Rows with 3, 4, 3, 2 hexagons
    dx = 1.5  # horizontal spacing for flat-top
    dy = math.sqrt(3)  # vertical spacing
    
    # Row 0: 3 hexagons
    for j in range(3):
        x = (j - 1) * dx
        y = 0
        configs.append((x, y, 0))
    
    # Row 1: 4 hexagons (offset)
    for j in range(4):
        x = (j - 1.5) * dx
        y = dy
        configs.append((x, y, 0))
    
    # Row 2: 3 hexagons
    for j in range(3):
        x = (j - 1) * dx
        y = 2 * dy
        configs.append((x, y, 0))
    
    # Row 3: 2 hexagons
    for j in range(2):
        x = (j - 0.5) * dx
        y = 3 * dy
        configs.append((x, y, 0))
    
    # Config 2: 2-3-4-3 pattern (shifted)
    configs2 = []
    for j in range(2):
        x = (j - 0.5) * dx
        y = 0
        configs2.append((x, y, 0))
    for j in range(3):
        x = (j - 1) * dx
        y = dy
        configs2.append((x, y, 0))
    for j in range(4):
        x = (j - 1.5) * dx
        y = 2 * dy
        configs2.append((x, y, 0))
    for j in range(3):
        x = (j - 1) * dx
        y = 3 * dy
        configs2.append((x, y, 0))
    configs.append(configs2)
    
    # Config 3: 4-4-4 grid (rectangular)
    configs3 = []
    for i in range(4):
        for j in range(3):
            x = (i - 1.5) * dx
            y = (j - 1.0) * dy
            configs3.append((x, y, 0))
    configs.append(configs3)
    
    # Config 4: Hexagonal ring + center
    configs4 = []
    configs4.append((0, 0, 0))  # center
    r1 = 1.732  # sqrt(3)
    for k in range(6):
        theta = math.radians(k * 60)
        configs4.append((r1 * math.cos(theta), r1 * math.sin(theta), 0))
    r2 = 3.464  # 2*sqrt(3)
    for k in range(5):
        theta = math.radians(30 + k * 72)
        configs4.append((r2 * math.cos(theta), r2 * math.sin(theta), 0))
    configs.append(configs4)
    
    # Config 5: 3-3-3-3 grid
    configs5 = []
    for i in range(4):
        for j in range(3):
            x = (i - 1.5) * dx
            y = (j - 1.0) * dy * 1.1  # slightly compressed vertically
            configs5.append((x, y, 0))
    configs.append(configs5)
    
    return configs

def compute_min_side_for_config(inner_positions, angle=0.0):
    """Compute minimum outer side length to contain all inner hexagons."""
    if not inner_positions:
        return 1.0
    
    # For each inner hexagon, find max distance from center in direction of outer hexagon vertices
    # Outer hexagon with side S and angle alpha has vertices at distance S from center
    # at angles alpha + k*60
    
    # We need S such that all inner hexagon vertices are inside
    # A point (px, py) is inside outer hexagon (center 0,0, side S, angle alpha) iff
    # for all 6 edge normals, the projection is within bounds
    
    # Simpler: for each inner hexagon vertex, compute required S
    # Point (px, py) is inside hexagon with side S, angle alpha iff:
    # max over 6 directions of |projection| <= S * cos(30°) = S * sqrt(3)/2
    
    max_required = 0.0
    for x, y, th in inner_positions:
        verts = hex_vertices(x, y, 1.0, th)
        for px, py in verts:
            # Check against outer hexagon with angle=angle
            a = math.radians(angle)
            # 6 edge normals
            for k in range(6):
                theta = a + k * math.pi / 3.0
                nx, ny = -math.sin(theta), math.cos(theta)
                proj = nx * px + ny * py
                # Required S: proj <= S * sqrt(3)/2
                required_S = abs(proj) / (math.sqrt(3) / 2)
                max_required = max(max_required, required_S)
    
    return max_required

def optimize_config_slsqp(initial_positions, angle=0.0):
    """Use SLSQP with hard constraints to optimize positions and find min S."""
    if time.time() > DEADLINE - 5:
        return None, None
    
    n_params = 3 * N  # x, y, angle for each inner hex
    
    def objective(params):
        if time.time() > DEADLINE - 2:
            return 1e10
        # Objective: minimize S (computed from containment)
        positions = []
        for i in range(N):
            positions.append((params[i*3], params[i*3+1], params[i*3+2]))
        return compute_min_side_for_config(positions, angle)
    
    def constraint_containment(params):
        """All inner hexagons must be inside outer hexagon."""
        S = compute_min_side_for_config([(params[i*3], params[i*3+1], params[i*3+2]) for i in range(N)], angle)
        # This is tricky - S is computed from params, so constraint is always satisfied by definition
        # Instead, we need to ensure no overlaps
        inner_polys = [hex_vertices(params[i*3], params[i*3+1], 1.0, params[i*3+2]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                # Return negative if overlapping (penalty)
                # Use SAT to check
                A, B = inner_polys[i], inner_polys[j]
                for poly in [A, B]:
                    for idx in range(len(poly)):
                        x1, y1 = poly[idx]
                        x2, y2 = poly[(idx+1) % len(poly)]
                        dx, dy = x2-x1, y2-y1
                        nx, ny = -dy, dx
                        norm = math.sqrt(nx*nx + ny*ny)
                        if norm < 1e-9:
                            continue
                        nx, ny = nx/norm, ny/norm
                        minA = min(nx*px + ny*py for px, py in A)
                        maxA = max(nx*px + ny*py for px, py in A)
                        minB = min(nx*px + ny*py for px, py in B)
                        maxB = max(nx*px + ny*py for px, py in B)
                        if maxA < minB - 1e-6 or maxB < minA - 1e-6:
                            break
                    else:
                        # No separating axis found - overlap
                        return -1.0
        return 1.0
    
    # Initial params
    x0 = np.zeros(n_params)
    for i in range(N):
        x0[i*3] = initial_positions[i][0]
        x0[i*3+1] = initial_positions[i][1]
        x0[i*3+2] = initial_positions[i][2]
    
    # Bounds: keep positions reasonable
    bounds = [(-5, 5)] * n_params
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      options={'maxiter': 100, 'ftol': 1e-4})
        if res.success or res.fun < 1e8:
            return res.x, res.fun
    except Exception:
        pass
    
    return None, None

def construct():
    t0 = time.time()
    best_side = float('inf')
    best_inner = None
    best_angle = 0.0
    
    # Try multiple topologies
    topologies = generate_hex_lattice_configs()
    
    for topo_idx, positions in enumerate(topologies):
        if time.time() > DEADLINE - 10:
            break
        
        # Normalize: center the configuration
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        centered = [(x - cx, y - cy, th) for x, y, th in positions]
        
        # Try with and without optimization
        for use_opt in [False, True]:
            if time.time() > DEADLINE - 5:
                break
            
            if use_opt:
                params, S = optimize_config_slsqp(centered)
                if params is None:
                    continue
                inner = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(N)]
            else:
                inner = centered
                S = compute_min_side_for_config(inner, 0.0)
            
            # Validate
            if validate_config([p for x,y,z in inner for p in (x,y,z)], (0,0), S, 0.0):
                if S < best_side:
                    best_side = S
                    best_inner = inner
                    best_angle = 0.0
    
    # If no valid config found, use fallback
    if best_inner is None:
        # Simple grid fallback
        dx = 1.5
        dy = math.sqrt(3)
        best_inner = []
        for i in range(4):
            for j in range(3):
                x = (i - 1.5) * dx
                y = (j - 1.0) * dy
                best_inner.append((x, y, 0.0))
        best_side = compute_min_side_for_config(best_inner, 0.0)
        best_angle = 0.0
    
    # Final validation
    if not validate_config([p for x,y,z in best_inner for p in (x,y,z)], (0,0), best_side, best_angle):
        # Emergency fallback: very conservative grid
        dx = 2.0
        dy = 2.0
        best_inner = []
        for i in range(4):
            for j in range(3):
                x = (i - 1.5) * dx
                y = (j - 1.0) * dy
                best_inner.append((x, y, 0.0))
        best_side = compute_min_side_for_config(best_inner, 0.0) * 1.1  # safety margin
        best_angle = 0.0
    
    outer_center = (0.0, 0.0)
    return best_inner, outer_center, best_side, best_angle
