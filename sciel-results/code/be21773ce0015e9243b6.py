"""Unit regular hexagons inside a regular hexagon n=12."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from math import cos, sin, pi, sqrt

N = 12
DEADLINE = time.time() + 140  # 140s to be safe under 150s limit

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * pi / 3
        verts.append([cx + side * cos(theta), cy + side * sin(theta)])
    return np.array(verts)

def hexagon_corners(cx, cy, side=1.0, angle_deg=0.0):
    """Return 6 corner points of a unit hexagon at (cx,cy) with given angle."""
    angle_rad = np.radians(angle_deg)
    corners = []
    for i in range(6):
        theta = angle_rad + i * pi / 3
        corners.append([cx + side * cos(theta), cy + side * sin(theta)])
    return np.array(corners)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside the hexagon centered at (cx,cy)."""
    angle_rad = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate point into hexagon's frame
    cos_a = cos(angle_rad)
    sin_a = sin(angle_rad)
    rx = cos_a * dx + sin_a * dy
    ry = -sin_a * dx + cos_a * dy
    # For a regular hexagon with side s, the apothem is s * sqrt(3)/2
    apothem = side * sqrt(3) / 2
    # Check: for each of the 6 faces, the distance from center projected onto the normal must be <= apothem
    for i in range(6):
        theta = i * pi / 3
        nx = cos(theta)
        ny = sin(theta)
        proj = rx * nx + ry * ny
        if proj > apothem + 1e-12:
            return False
    return True

def hexagons_disjoint(i1, j1, a1, i2, j2, a2):
    """Check if two unit hexagons are disjoint (non-overlapping).
    Hexagon 1: center (i1, j1), angle a1
    Hexagon 2: center (i2, j2), angle a2
    """
    # Two convex polygons are disjoint if no edge of one intersects any edge of the other
    # and neither contains the other.
    # For regular hexagons of same size, check if any vertex of one is inside the other,
    # and if any edges cross.
    c1 = hexagon_corners(i1, j1, 1.0, a1)
    c2 = hexagon_corners(i2, j2, 1.0, a2)
    
    # Check if any vertex of hex1 is inside hex2
    for v in c1:
        if point_in_hexagon(v[0], v[1], i2, j2, 1.0, a2):
            return False
    # Check if any vertex of hex2 is inside hex1
    for v in c2:
        if point_in_hexagon(v[0], v[1], i1, j1, 1.0, a1):
            return False
    # Check edge crossings
    for i in range(6):
        p1 = c1[i]
        p2 = c1[(i+1) % 6]
        for j in range(6):
            p3 = c2[j]
            p4 = c2[(j+1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    return True

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    # Handle collinear cases
    if abs(d1) < 1e-12 and on_segment(p3, p4, p1):
        return True
    if abs(d2) < 1e-12 and on_segment(p3, p4, p2):
        return True
    if abs(d3) < 1e-12 and on_segment(p1, p2, p3):
        return True
    if abs(d4) < 1e-12 and on_segment(p1, p2, p4):
        return True
    return False

def on_segment(p, q, r):
    """Check if point q lies on segment p-r (assuming collinear)."""
    return (min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12 and
            min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12)

def min_outer_side(inner_positions, outer_center, outer_angle):
    """Find the minimum side length of outer hexagon (centered at outer_center, rotated by outer_angle)
    that contains all inner hexagons.
    inner_positions: list of (x, y, angle_deg)
    """
    ox, oy = outer_center
    # The minimum enclosing side is determined by the maximum distance from center
    # to any vertex of any inner hexagon, measured in the outer hexagon's frame.
    max_dist = 0
    for (ix, iy, ia) in inner_positions:
        corners = hexagon_corners(ix, iy, 1.0, ia)
        for v in corners:
            # Transform to outer frame
            dx = v[0] - ox
            dy = v[1] - oy
            angle_rad = np.radians(outer_angle)
            cos_a = cos(angle_rad)
            sin_a = sin(angle_rad)
            rx = cos_a * dx + sin_a * dy
            ry = -sin_a * dx + cos_a * dy
            # The maximum "radius" in a regular hexagon direction
            for i in range(6):
                theta = i * pi / 3
                proj = abs(rx * cos(theta) + ry * sin(theta))
                if proj > max_dist:
                    max_dist = proj
    # The side length of the outer hexagon is max_dist / (sqrt(3)/2) = max_dist * 2/sqrt(3)
    side = max_dist * 2.0 / sqrt(3)
    return side

def is_valid_configuration(inner_positions, outer_center, outer_side, outer_angle):
    """Check if configuration is valid: all inner hexagons disjoint and inside outer hexagon."""
    # Check disjointness
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            x1, y1, a1 = inner_positions[i]
            x2, y2, a2 = inner_positions[j]
            if not hexagons_disjoint(x1, y1, a1, x2, y2, a2):
                return False
    # Check containment
    ox, oy = outer_center
    for (ix, iy, ia) in inner_positions:
        corners = hexagon_corners(ix, iy, 1.0, ia)
        for v in corners:
            if not point_in_hexagon(v[0], v[1], ox, oy, outer_side, outer_angle):
                return False
    return True

def objective(params, inner_positions):
    """Objective: minimize outer side length given fixed inner positions."""
    cx, cy, angle = params
    return min_outer_side(inner_positions, (cx, cy), angle)

def objective_with_validity(params, inner_positions):
    """Objective with penalty for invalid configurations."""
    cx, cy, angle = params
    side = min_outer_side(inner_positions, (cx, cy), angle)
    # Check if all inner hexagons are inside the outer hexagon of this side
    valid = True
    for (ix, iy, ia) in inner_positions:
        corners = hexagon_corners(ix, iy, 1.0, ia)
        for v in corners:
            if not point_in_hexagon(v[0], v[1], cx, cy, side, angle):
                valid = False
                break
        if not valid:
            break
    if not valid:
        return side + 100.0
    # Check disjointness
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            x1, y1, a1 = inner_positions[i]
            x2, y2, a2 = inner_positions[j]
            if not hexagons_disjoint(x1, y1, a1, x2, y2, a2):
                return side + 50.0
    return side

def generate_lattice_config(spacing, angle_deg, offset_x, offset_y):
    """Generate a grid/lattice of hexagons."""
    positions = []
    # Try a hexagonal lattice
    # For 12 hexagons, try various arrangements
    arrangements = [
        # 3x4 grid
        lambda k: (3 * (k % 4), spacing * (k // 4)) if k < 12 else None,
        # 4x3 grid
        lambda k: (4 * (k % 3), spacing * (k // 3)) if k < 12 else None,
        # Hexagonal arrangement (1+6+... pattern doesn't fit 12 well)
        # Row-based hex packing
    ]
    
    # Simpler: generate points on a hexagonal grid
    # Hex grid spacing: horizontal = spacing*sqrt(3), vertical = spacing*1.5
    hx = spacing * sqrt(3)
    hy = spacing * 1.5
    points = []
    for row in range(-4, 5):
        for col in range(-4, 5):
            x = col * hx + (hx / 2 if row % 2 else 0)
            y = row * hy
            points.append((x + offset_x, y + offset_y, angle_deg))
    
    # Select 12 points that form a compact cluster
    # Sort by distance from center
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    return points[:N]

def generate_random_config(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    positions = []
    for _ in range(N):
        x = rng.uniform(-5, 5)
        y = rng.uniform(-5, 5)
        a = rng.uniform(0, 60)  # Hexagon has 6-fold symmetry, 0-60 is enough
        positions.append([x, y, a])
    return positions

def optimize_inner_positions(inner_positions, outer_params, max_iters=200):
    """Optimize inner positions to minimize outer side while keeping them disjoint and inside."""
    cx, cy, angle, side = outer_params
    
    def inner_obj(params):
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        s = min_outer_side(positions, (cx, cy), angle)
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(positions[i][0], positions[i][1], positions[i][2],
                                         positions[j][0], positions[j][1], positions[j][2]):
                    return s + 10.0
        # Check containment
        for p in positions:
            corners = hexagon_corners(p[0], p[1], 1.0, p[2])
            for v in corners:
                if not point_in_hexagon(v[0], v[1], cx, cy, s, angle):
                    return s + 10.0
        return s
    
    params0 = np.array([p for pos in inner_positions for p in pos])
    bounds = [(-10, 10)] * (3 * N)
    
    try:
        res = minimize(inner_obj, params0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iters, 'ftol': 1e-12})
        if res.fun < side:
            new_positions = []
            for i in range(N):
                new_positions.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2]])
            return new_positions, res.fun
    except:
        pass
    return inner_positions, side

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # Phase 1: Generate diverse initial configurations
    configs = []
    
    # Lattice-based configurations with various spacings and rotations
    for spacing in [1.9, 2.0, 2.1, 2.2, 2.3]:
        for angle in [0, 15, 30, 45, 60]:
            for ox in [-1, 0, 1]:
                for oy in [-1, 0, 1]:
                    if time.time() > DEADLINE - 60:
                        break
                    config = generate_lattice_config(spacing, angle, ox, oy)
                    if len(config) >= N:
                        configs.append(config)
    
    # Random configurations
    for seed in range(20):
        if time.time() > DEADLINE - 60:
            break
        configs.append(generate_random_config(seed))
    
    # Phase 2: For each config, optimize outer hexagon parameters
    for inner in configs:
        if time.time() > DEADLINE - 40:
            break
        
        # Optimize outer center and angle
        bounds = [(-5, 5), (-5, 5), (0, 60)]
        
        try:
            res = differential_evolution(
                objective, bounds, args=(inner,),
                maxiter=30, tol=1e-8, seed=42, polish=True,
                init='sobol'
            )
            if res.fun < best_score:
                cx, cy, angle = res.x
                best_score = res.fun
                best_result = (inner, (cx, cy), best_score, angle)
        except:
            pass
    
    # Phase 3: Optimize inner positions for the best configuration
    if best_result is not None and time.time() < DEADLINE - 30:
        inner_best = best_result[0]
        outer_best = best_result
        for _ in range(3):
            if time.time() > DEADLINE - 20:
                break
            inner_best, new_side = optimize_inner_positions(
                inner_best, (outer_best[1][0], outer_best[1][1], outer_best[3], outer_best[2]),
                max_iters=100
            )
            if new_side < best_score:
                best_score = new_side
                best_result = (inner_best, outer_best[1], best_score, outer_best[3])
    
    # Phase 4: Final polish of outer parameters
    if best_result is not None and time.time() < DEADLINE - 10:
        inner_final = best_result[0]
        cx0, cy0 = best_result[1]
        angle0 = best_result[3]
        try:
            res = minimize(
                objective, [cx0, cy0, angle0], args=(inner_final,),
                method='L-BFGS-B',
                bounds=[(-5, 5), (-5, 5), (0, 60)],
                options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-14}
            )
            if res.fun < best_score:
                cx, cy, angle = res.x
                best_score = res.fun
                best_result = (inner_final, (cx, cy), best_score, angle)
        except:
            pass
    
    if best_result is None:
        # Fallback: very large hexagon with hexagons in a line
        return [[2.0 * k, 0.0, 0.0] for k in range(-6, 6)], (0.0, 0.0), 40.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
