import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _separating_axis_distance(poly1, poly2):
    """
    Compute the minimum translation distance to separate two convex polygons.
    Returns 0 if they overlap (or touch), positive if separated.
    Uses the separating axis theorem.
    """
    # Get all edge normals from both polygons
    axes = []
    for poly in [poly1, poly2]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            # Normal to the edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal /= norm
                axes.append(normal)
    
    min_dist = float('inf')
    for axis in axes:
        # Project both polygons onto this axis
        proj1 = np.dot(poly1, axis)
        proj2 = np.dot(poly2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Distance between intervals
        if max1 < min2:
            dist = min2 - max1
        elif max2 < min1:
            dist = min1 - max2
        else:
            dist = 0.0
        min_dist = min(min_dist, dist)
    
    return min_dist

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside a regular hexagon."""
    # Transform to hexagon's local coordinate system
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    rx = dx * np.cos(-angle) - dy * np.sin(-angle)
    ry = dx * np.sin(-angle) + dy * np.cos(-angle)
    
    # In local coords, hexagon has vertices at angle 0, 60, 120, 180, 240, 300
    # The hexagon is defined by |x| <= side, |y| <= side*sqrt(3)/2, and |x|+|y|/sqrt(3) <= side
    # Actually, let's use the standard form: for a hexagon with flat top, 
    # the constraints are different. Let me use the vertex-based approach.
    
    # For a hexagon centered at origin with vertices at angles 0,60,...,300 and circumradius = side:
    # The half-width in direction theta is side * cos(theta - nearest_vertex_angle) / cos(30deg)
    # Simpler: check against all 6 edge half-planes
    
    verts = []
    for i in range(6):
        theta = i * np.pi / 3
        vx = side * np.cos(theta)
        vy = side * np.sin(theta)
        verts.append((vx, vy))
    verts = np.array(verts)
    
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        # Check if point is on the same side as the center (origin)
        dot_edge = np.dot(rx, normal)
        dot_ref = np.dot(0.0, normal)  # 0
        # Actually, the inward normal should point toward the center
        # Let's compute the cross product to determine orientation
        # For CCW polygon, the interior is to the left of each edge
        cross = edge[0] * (ry - p1[1]) - edge[1] * (rx - p1[0])
        if cross < -1e-9:
            return False
    return True

def _max_violation(params, outer_side):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_cx, outer_cy, outer_side, outer_angle]
    Returns a penalty value: 0 if valid, positive if invalid.
    """
    n = N
    inner_params = params[:3 * n]
    outer_cx = params[3 * n]
    outer_cy = params[3 * n + 1]
    outer_side_val = params[3 * n + 2]
    outer_angle = params[3 * n + 3]
    
    # Extract inner hexagons
    hexes = []
    for i in range(n):
        x = inner_params[3 * i]
        y = inner_params[3 * i + 1]
        a = inner_params[3 * i + 2]
        hexes.append((x, y, a))
    
    penalty = 0.0
    
    # Check container violations
    outer_verts = _hex_vertices(outer_cx, outer_cy, outer_side_val, outer_angle)
    for (x, y, a) in hexes:
        inner_verts = _hex_vertices(x, y, 1.0, a)
        for vx, vy in inner_verts:
            if not _point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side_val, outer_angle):
                # Compute how far outside
                # Simplified: just add a penalty
                penalty += 1.0
    
    # Check pairwise overlaps
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = hexes[i]
            x2, y2, a2 = hexes[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            dist = _separating_axis_distance(v1, v2)
            if dist < 0.01:  # Overlapping or too close
                penalty += max(0, 0.01 - dist)
    
    return penalty

def construct():
    n = N
    # Initial guess: place hexagons in a grid-like pattern
    # For a hexagonal packing, the optimal arrangement for small N is hexagonal lattice
    
    # Start with a reasonable initial configuration
    # Hexagonal lattice spacing: for unit hexagons, center-to-center distance is 2*sqrt(3)/sqrt(3) = 2 for adjacent
    # Actually, for two unit hexagons touching, center distance = 2 * (apothem) = 2 * (sqrt(3)/2) = sqrt(3)
    # Wait, apothem of unit hexagon = sqrt(3)/2, so touching distance = 2 * sqrt(3)/2 = sqrt(3)
    
    # Let's use a hexagonal grid
    positions = []
    # Center at origin, then surrounding rings
    # For N=11: 1 center + 6 first ring + 4 from second ring (or some arrangement)
    
    # Simple approach: linear arrangement won't work well. Use hexagonal packing.
    # Hexagonal lattice: positions at (i*a + j*a/2, j*a*sqrt(3)/2) where a = sqrt(3) for touching
    a = np.sqrt(3)  # Center distance for touching unit hexagons
    
    # Generate positions in a hexagonal pattern
    # For N=11, use a 3-3-3-2 or similar arrangement
    # Let's just use a compact hexagonal cluster
    
    # Better: use a predefined compact arrangement
    # For 11 hexagons, a good arrangement is a hexagon of radius 2 minus corners
    # Or just use differential evolution with good initial bounds
    
    # Initial positions: spread out in a hexagonal pattern
    init_positions = []
    # Ring 0: center
    init_positions.append((0, 0))
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        init_positions.append((a * np.cos(angle), a * np.sin(angle)))
    # Ring 2: need 4 more
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        if len(init_positions) < n:
            init_positions.append((2 * a * np.cos(angle), 2 * a * np.sin(angle)))
    
    # Pad if needed
    while len(init_positions) < n:
        init_positions.append((0, 0))
    
    # Truncate if needed
    init_positions = init_positions[:n]
    
    # Initial angles: all 0
    # Outer hexagon: start with a large enough side
    # The minimum side for N hexagons in hexagonal packing...
    # For a hexagonal cluster of radius r (in terms of hexagon steps), 
    # the circumradius is roughly r * a + 1 (for the outer hexagons)
    
    # Let's estimate: for 11 hexagons, a compact arrangement fits in a hexagon of side ~3-4
    outer_side_init = 5.0
    
    # Build initial params
    init_params = []
    for (x, y) in init_positions:
        init_params.extend([x, y, 0.0])
    init_params.extend([0.0, 0.0, outer_side_init, 0.0])
    
    # Bounds for differential evolution
    # Inner positions: roughly -10 to 10
    # Angles: 0 to 60 (hexagons have 60-degree symmetry)
    # Outer: center -10 to 10, side 2 to 10, angle 0 to 60
    
    bounds = []
    for i in range(n):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 60))    # angle
    
    bounds.append((-5, 5))   # outer cx
    bounds.append((-5, 5))   # outer cy
    bounds.append((2, 10))   # outer side
    bounds.append((0, 60))   # outer angle
    
    # Objective function for differential evolution
    def objective(params):
        penalty = _max_violation(params, params[3 * n + 2])
        # We want to minimize the outer side length, but also ensure validity
        # Use a weighted objective: outer_side + large_penalty * violation
        outer_side_val = params[3 * n + 2]
        return outer_side_val + 1000.0 * penalty
    
    # Run differential evolution with multiple starts
    best_result = None
    best_score = float('inf')
    
    n_starts = 5
    for start in range(n_starts):
        if start == 0:
            x0 = np.array(init_params)
        else:
            # Perturb initial positions
            x0 = np.array(init_params, dtype=float)
            rng = np.random.RandomState(start * 42 + 7)
            x0[:3 * n] += rng.uniform(-2, 2, size=3 * n)
            x0[3 * n:3 * n + 2] += rng.uniform(-1, 1, size=2)
            x0[3 * n + 2] = rng.uniform(3, 8)
            x0[3 * n + 3] = rng.uniform(0, 60)
        
        # Clip to bounds
        for i in range(len(bounds)):
            x0[i] = np.clip(x0[i], bounds[i][0], bounds[i][1])
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=200,
                popsize=15,
                tol=0.01,
                seed=start * 123 + 456,
                polish=True
            )
            
            if result.success or result.fun < best_score:
                params = result.x
                penalty = _max_violation(params, params[3 * n + 2])
                if penalty < 0.01:  # Valid solution
                    outer_side_val = params[3 * n + 2]
                    if outer_side_val < best_score:
                        best_score = outer_side_val
                        best_result = params
        except Exception as e:
            continue
    
    # If no valid solution found, use initial guess with a larger outer hexagon
    if best_result is None:
        # Fallback: use the initial arrangement with a large enough outer hexagon
        # Compute bounding box of initial positions
        xs = [p[0] for p in init_positions]
        ys = [p[1] for p in init_positions]
        max_extent = max(abs(x) for x in xs) + max(abs(y) for y in ys) + 2
        outer_side = max_extent * 2 / np.sqrt(3) + 2
        best_result = np.array(init_params)
        best_result[3 * n + 2] = outer_side
    
    # Extract solution
    params = best_result
    inner = []
    for i in range(n):
        x = params[3 * i]
        y = params[3 * i + 1]
        a = params[3 * i + 2]
        inner.append([float(x), float(y), float(a)])
    
    outer_cx = float(params[3 * n])
    outer_cy = float(params[3 * n + 1])
    outer_side = float(params[3 * n + 2])
    outer_angle = float(params[3 * n + 3])
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END
