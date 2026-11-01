"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def _hex_vertices_unit(cx, cy, angle_deg):
    """Unit side length hexagon vertices."""
    return _hex_vertices(cx, cy, 1.0, angle_deg)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    ca, sa = math.cos(-a), math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side s, the distance from center to each edge is s * sqrt(3)/2
    # The 6 normal directions are at angles 30, 90, 150, 210, 270, 330 degrees
    apothem = side * math.sqrt(3) / 2
    for i in range(6):
        na = math.radians(30 + i * 60)
        nx = math.cos(na)
        ny = math.sin(na)
        # The edge is at distance apothem in direction (nx, ny)
        # Point must satisfy nx*lx + ny*ly <= apothem
        if nx * lx + ny * ly > apothem + 1e-12:
            return False
    return True

def _hex_edge_distance(h1, h2):
    """
    Compute the minimum distance between two hexagons.
    h1, h2 are (cx, cy, angle_deg) with unit side length.
    Returns the minimum distance between their boundaries (negative if overlapping).
    """
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # First, quick check using center distance
    dc = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    # The maximum extent of a unit hexagon from center is 2 (to opposite vertex)
    # Minimum extent is sqrt(3)/2 (to edge)
    # If centers are very far, no overlap possible
    if dc > 4.0:
        return dc - 4.0
    
    # Check if any vertex of h1 is inside h2 or vice versa
    verts1 = _hex_vertices_unit(cx1, cy1, a1)
    verts2 = _hex_vertices_unit(cx2, cy2, a2)
    
    # Check vertices of h1 against h2
    for v in verts1:
        if _point_in_hex(v[0], v[1], cx2, cy2, 1.0, a2):
            # Overlapping, return negative distance
            # Find the minimum penetration depth
            # This is a heuristic; for exact computation we'd need more work
            # For now, return a small negative value
            return -0.1
    
    for v in verts2:
        if _point_in_hex(v[0], v[1], cx1, cy1, 1.0, a1):
            return -0.1
    
    # Compute minimum distance between edges
    min_dist = float('inf')
    for i in range(6):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % 6]
        for j in range(6):
            q1 = verts2[j]
            q2 = verts2[(j + 1) % 6]
            d = _seg_seg_dist(p1, p2, q1, q2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _seg_seg_dist(p1, p2, q1, q2):
    """Minimum distance between two line segments."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    
    denom = a * c - b * b
    if denom < 1e-14:
        # Parallel or degenerate
        s = 0.0
        t = 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    
    if s < 0:
        s = 0.0
    elif s > 1:
        s = 1.0
    if t < 0:
        t = 0.0
    elif t > 1:
        t = 1.0
    
    # Recompute with clamped values
    if s < 0:
        s = 0.0
    if s > 1:
        s = 1.0
    if t < 0:
        t = 0.0
    if t > 1:
        t = 1.0
    
    p = p1 + s * d1
    q = q1 + t * d2
    return np.linalg.norm(p - q)

def _outer_side_length(inner, outer_center, outer_angle):
    """
    Compute the minimum outer hexagon side length that contains all inner hexagons.
    inner: list of (cx, cy, angle_deg)
    outer_center: (cx, cy)
    outer_angle: degrees
    """
    ox, oy = outer_center
    oa = outer_angle
    
    # For each inner hexagon, find the maximum "extent" in each of the 6 outer hexagon directions
    # The outer hexagon with center (ox,oy), side S, angle oa has edges at distance S*sqrt(3)/2 from center
    # in directions 30+60k degrees (rotated by oa)
    
    # For containment, each vertex of each inner hex must be inside the outer hex
    # The constraint is: for each outer edge direction, the projection of each inner vertex
    # onto that direction must be <= S*sqrt(3)/2
    
    max_extent = 0.0
    for (cx, cy, ia) in inner:
        verts = _hex_vertices_unit(cx, cy, ia)
        for v in verts:
            # Transform to outer hex frame
            dx = v[0] - ox
            dy = v[1] - oy
            a = math.radians(oa)
            ca, sa = math.cos(-a), math.sin(-a)
            lx = ca * dx - sa * dy
            ly = sa * dx + ca * dy
            
            # Check against 6 edges
            for k in range(6):
                na = math.radians(30 + k * 60)
                nx = math.cos(na)
                ny = math.sin(na)
                proj = nx * lx + ny * ly
                if proj > max_extent:
                    max_extent = proj
    
    # max_extent is the maximum projection onto any edge normal
    # The apothem of the outer hex must be >= max_extent
    # apothem = S * sqrt(3) / 2
    S = max_extent * 2.0 / math.sqrt(3)
    return max(S, 1.0)  # At least 1.0

def _objective(params, inner_positions, outer_center, outer_angle):
    """
    Objective: minimize outer side length.
    params: array of [cx1, cy1, a1, cx2, cy2, a2, ...] for N inner hexagons
    """
    n = len(inner_positions)
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        a = params[3*i+2]
        inner.append((cx, cy, a))
    
    # Check for overlaps
    min_pair_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = _hex_edge_distance(inner[i], inner[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Penalty for overlaps
    penalty = 0.0
    if min_pair_dist < 0.01:
        penalty = (0.01 - min_pair_dist) * 100.0
    
    S = _outer_side_length(inner, outer_center, outer_angle)
    return S + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: hexagonal packing
    # Place N hexagons in a hexagonal arrangement
    # For N=11, use a 1-2-3-2-1 pattern or similar
    
    # Simple initial: all at origin with slight offsets
    # Better: use known good packing
    
    # Hexagonal lattice positions for unit hexagons
    # Adjacent hexagons in hex lattice are separated by sqrt(3) in the direction of edges
    # and by 2 in the direction of vertices
    
    # Let's use a compact hexagonal arrangement
    # For N hexagons, optimal arrangement is roughly hexagonal
    
    # Generate hexagonal lattice positions
    def hex_lattice_positions(n):
        """Generate n positions in a hexagonal lattice."""
        positions = []
        # Ring 0: center
        positions.append((0.0, 0.0))
        count = 1
        ring = 1
        while count < n:
            # Ring r has 6*r positions
            for i in range(6 * ring):
                if count >= n:
                    break
                # Position on ring r, index i
                side = i // ring  # Which side of the hexagon
                pos_on_side = i % ring
                if side == 0:
                    # Top-right edge
                    x = ring * math.sqrt(3)/2 + pos_on_side * math.sqrt(3)/2
                    y = ring * 0.5 + pos_on_side * 0.5
                elif side == 1:
                    # Bottom-right
                    x = ring * math.sqrt(3)/2 - pos_on_side * math.sqrt(3)/2
                    y = -ring * 0.5 - pos_on_side * 0.5
                elif side == 2:
                    # Bottom
                    x = -pos_on_side * math.sqrt(3)/2
                    y = -ring
                elif side == 3:
                    # Bottom-left
                    x = -ring * math.sqrt(3)/2 - pos_on_side * math.sqrt(3)/2
                    y = -ring * 0.5 + pos_on_side * 0.5
                elif side == 4:
                    # Top-left
                    x = -ring * math.sqrt(3)/2 + pos_on_side * math.sqrt(3)/2
                    y = ring * 0.5 - pos_on_side * 0.5
                else:
                    # Top
                    x = pos_on_side * math.sqrt(3)/2
                    y = ring
                positions.append((x, y))
                count += 1
            ring += 1
        return positions[:n]
    
    # For hexagons, the lattice spacing is different
    # Two unit hexagons that share an edge have centers separated by sqrt(3)
    # in the direction perpendicular to the shared edge
    
    # Let me use a simpler approach: place them in a grid
    # Hexagonal tiling: each hex has 6 neighbors at distance sqrt(3)
    
    # Actually, for the initial guess, let's just use a line or small cluster
    # and let the optimizer do the work
    
    # Initial: place in a hexagonal cluster
    # Center + 6 around + 4 more
    def initial_config(n):
        config = []
        # Center
        config.append([0.0, 0.0, 0.0])
        # 6 neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
        for i in range(6):
            a = i * math.pi / 3
            x = math.sqrt(3) * math.cos(a)
            y = math.sqrt(3) * math.sin(a)
            config.append([x, y, 0.0])
        # 4 more: next ring
        # Positions at distance 2*sqrt(3) in directions 30, 90, 150, 210, 270, 330
        # Or at distance sqrt(3) in directions 30, 90, ... but those are already taken
        # Next ring: 6 positions at distance 2*sqrt(3) in directions 0, 60, ...
        # Plus 6 at distance sqrt(3)*sqrt(3)=3 in directions 30, 90, ...
        # Let's just add 4 more at reasonable positions
        extra_positions = [
            (2*math.sqrt(3), 0),
            (math.sqrt(3), math.sqrt(3)),
            (-math.sqrt(3), math.sqrt(3)),
            (-2*math.sqrt(3), 0),
        ]
        for i in range(min(4, n - 7)):
            x, y = extra_positions[i]
            config.append([x, y, 0.0])
        
        return config[:n]
    
    init = initial_config(N)
    
    # Optimize using differential evolution first (global), then local polish
    # Parameters: [cx1, cy1, a1, cx2, cy2, a2, ..., cxN, cyN, aN]
    
    def obj_func(params):
        return _objective(params, init, (0.0, 0.0), 0.0)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))  # cx
        bounds.append((-10, 10))  # cy
        bounds.append((-180, 180))  # angle
    
    # Use differential evolution for global optimization
    best_S = float('inf')
    best_params = None
    
    # Try a few different initial configurations
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        
        if trial == 0:
            init_params = np.array([v for hex in init for v in hex])
        else:
            # Add some randomness
            init_params = np.array([v + np.random.randn() * 0.5 for hex in init for v in hex])
        
        try:
            result = differential_evolution(
                obj_func, bounds, 
                maxiter=200, popsize=15,
                tol=1e-4, seed=42+trial,
                polish=True
            )
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([v for hex in init for v in hex])
        best_S = obj_func(best_params)
    
    # Local polish with Nelder-Mead or L-BFGS-B
    if time.time() < deadline - 10:
        try:
            result = minimize(
                obj_func, best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 5:
        try:
            result = minimize(
                obj_func, best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract final configuration
    inner = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([cx, cy, a])
    
    # Compute the actual outer side length
    outer_side = _outer_side_length(inner, (0.0, 0.0), 0.0)
    
    return inner, (0.0, 0.0), outer_side, 0.0

# EVOLVE-BLOCK-END
