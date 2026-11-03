"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Place 12 disjoint regular hexagons of side length 1 inside a regular hexagon,
    minimizing the outer side length.
    
    Returns: (inner, center, side, angle)
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (x, y) of outer hexagon center
      side: side length of outer hexagon
      angle: rotation angle of outer hexagon in degrees
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    def hex_vertices(cx, cy, s, angle_deg):
        """Return vertices of a regular hexagon centered at (cx, cy) with side s and rotation angle_deg."""
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3.0
            verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
        return verts
    
    def point_in_hex(px, py, cx, cy, s, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (cx, cy) with side s and rotation angle_deg."""
        # Transform point to hexagon's local frame
        dx = px - cx
        dy = py - cy
        angle = math.radians(-angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
        # The hexagon is the intersection of 6 half-planes
        # For a regular hexagon with circumradius = s (side length = circumradius for regular hex)
        # Actually for regular hexagon, side length = circumradius
        # The support function in direction theta is s * cos(0) for directions aligned with vertices
        # Let's use the standard test: a point is inside if for all 6 edge normals, the dot product <= s
        # Edge normals point outward at angles 30, 90, 150, 210, 270, 330 degrees (perpendicular to edges)
        # Wait, let me think more carefully.
        # A regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (circumradius s)
        # has edges between consecutive vertices. The outward normal of each edge is perpendicular to the edge.
        # Edge from vertex 0 to vertex 1: midpoint direction is 30 degrees, so normal is at 30 degrees.
        # The distance from center to each edge (apothem) is s * cos(30) = s * sqrt(3)/2
        # So point is inside if for each of the 6 edge normals at angles 30+60*k, 
        # the dot product of local point with unit normal <= s*sqrt(3)/2
        apothem = s * math.sqrt(3) / 2.0
        for k in range(6):
            normal_angle = math.radians(30 + 60 * k)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            if lx * nx + ly * ny > apothem + 1e-10:
                return False
        return True
    
    def hex_overlap_distance(h1, h2):
        """
        Compute a measure of overlap between two hexagons.
        h1, h2 are (cx, cy, s, angle_deg).
        Returns 0 if disjoint, positive if overlapping.
        Uses sampling of boundary points.
        """
        cx1, cy1, s1, a1 = h1
        cx2, cy2, s2, a2 = h2
        
        # Sample points on each hexagon's boundary and check if inside the other
        # Also check if centers are inside the other hexagon
        # For a robust check, we'll sample multiple points
        
        # First, check center of hex1 in hex2
        if point_in_hex(cx1, cy1, cx2, cy2, s2, a2):
            # Center of h1 is inside h2, so they overlap
            # Estimate penetration depth
            return 1.0
        
        if point_in_hex(cx2, cy2, cx1, cy1, s1, a1):
            return 1.0
        
        # Sample boundary points of hex1
        n_samples = 24
        max_pen = 0.0
        for i in range(n_samples):
            theta = 2 * math.pi * i / n_samples
            # Point on boundary of hex1
            # The boundary of a regular hexagon with circumradius s1 at angle theta
            # is at distance r(theta) from center
            # For a regular hexagon, the radial distance varies between apothem and circumradius
            # Let's just sample vertices and edge midpoints
            pass
        
        # Better approach: check if any vertex of h1 is inside h2, or any vertex of h2 is inside h1
        v1 = hex_vertices(cx1, cy1, s1, a1)
        v2 = hex_vertices(cx2, cy2, s2, a2)
        
        for vx, vy in v1:
            if point_in_hex(vx, vy, cx2, cy2, s2, a2):
                return 1.0
        
        for vx, vy in v2:
            if point_in_hex(vx, vy, cx1, cy1, s1, a1):
                return 1.0
        
        # Check edge intersections (simplified: if no vertex is inside and no edge crosses, they're disjoint)
        # For simplicity, if no vertices are inside, assume disjoint (may miss some cases but good enough for optimization)
        return 0.0
    
    def outer_hex_contains_inner(outer, inner_hex):
        """Check if outer hexagon contains the inner hexagon."""
        ocx, ocy, os, oangle = outer
        icx, icy, is_, iangle = inner_hex
        
        # Check all vertices of inner hex are inside outer hex
        iv = hex_vertices(icx, icy, is_, iangle)
        for vx, vy in iv:
            if not point_in_hex(vx, vy, ocx, ocy, os, oangle):
                return False
        return True
    
    def compute_objective(params, n_inner=12):
        """
        Compute the minimum outer hexagon side length needed to contain all inner hexagons.
        params: flat array of [x1, y1, a1, x2, y2, a2, ..., xn, yn, an]
        Returns: (side_length, success)
        """
        if time.time() - start_time > deadline:
            return (float('inf'), False)
        
        inner_hexes = []
        for i in range(n_inner):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_hexes.append((x, y, 1.0, a))
        
        # Check pairwise non-overlap
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                if hex_overlap_distance(inner_hexes[i], inner_hexes[j]) > 0:
                    return (float('inf'), False)
        
        # Find minimum enclosing hexagon
        # We'll try different rotations of the outer hexagon and find the minimum side length
        best_side = float('inf')
        
        # The outer hexagon should be centered at the centroid of inner hexagons
        cx = np.mean([h[0] for h in inner_hexes])
        cy = np.mean([h[1] for h in inner_hexes])
        
        # Try several outer hexagon rotations
        for oangle_deg in range(0, 60, 5):
            oangle = math.radians(oangle_deg)
            
            # For each inner hexagon, compute the required outer side length
            # The outer hexagon with side S and rotation oangle contains a point (px, py)
            # iff in the rotated frame, the point satisfies the half-plane constraints
            # The maximum "support" value over all inner hexagon vertices gives the required S
            
            max_support = 0.0
            for ih in inner_hexes:
                iv = hex_vertices(ih[0], ih[1], ih[2], ih[3])
                for vx, vy in iv:
                    # Transform to outer hexagon's frame
                    dx = vx - cx
                    dy = vy - cy
                    cos_a = math.cos(-oangle)
                    sin_a = math.sin(-oangle)
                    lx = dx * cos_a - dy * sin_a
                    ly = dx * sin_a + dy * cos_a
                    
                    # In local frame, the hexagon with circumradius S has apothem S*sqrt(3)/2
                    # Point is inside if for all 6 edge normals, dot product <= S*sqrt(3)/2
                    # The required S is max over all points and all 6 normals of (dot product) / (sqrt(3)/2)
                    for k in range(6):
                        normal_angle = math.radians(30 + 60 * k)
                        nx = math.cos(normal_angle)
                        ny = math.sin(normal_angle)
                        support = lx * nx + ly * ny
                        required_s = support / (math.sqrt(3) / 2.0)
                        if required_s > max_support:
                            max_support = required_s
            
            # Add a small margin for numerical safety
            if max_support < best_side:
                best_side = max_support
        
        return (best_side, True)
    
    # Initial configuration: place hexagons in a hexagonal pattern
    # For 12 hexagons, a good starting arrangement is a central hexagon surrounded by others
    # or a 3-4-3 or similar pattern
    
    # Let's try a hexagonal close-packed arrangement
    # Center at origin, then rings
    # For 12 hexagons: 1 center + 6 in first ring + 5 in second ring (partial)
    # Or: 2-3-2-3-2 pattern in rows
    
    # Simple initial guess: grid-like arrangement
    initial_params = []
    spacing = 1.75  # slightly more than 2*1 to avoid overlap initially
    
    # Arrange in a hexagonal pattern
    positions = [
        (0, 0),
        (spacing, 0),
        (-spacing, 0),
        (spacing/2, spacing*math.sqrt(3)/2),
        (-spacing/2, spacing*math.sqrt(3)/2),
        (spacing/2, -spacing*math.sqrt(3)/2),
        (-spacing/2, -spacing*math.sqrt(3)/2),
        (1.5*spacing, spacing*math.sqrt(3)/2),
        (1.5*spacing, -spacing*math.sqrt(3)/2),
        (-1.5*spacing, spacing*math.sqrt(3)/2),
        (-1.5*spacing, -spacing*math.sqrt(3)/2),
        (0, 2*spacing*math.sqrt(3)/2),
    ]
    
    for x, y in positions:
        initial_params.extend([x, y, 0.0])
    
    initial_params = np.array(initial_params)
    
    # Use Nelder-Mead optimization to minimize the outer hexagon side length
    best_params = initial_params.copy()
    best_side, success = compute_objective(initial_params)
    
    if not success or best_side == float('inf'):
        # Try to fix overlaps by spreading out
        for i in range(N):
            initial_params[3*i] += (i % 3 - 1) * 0.5
            initial_params[3*i+1] += (i // 3 - 2) * 0.5
        best_params = initial_params.copy()
        best_side, success = compute_objective(initial_params)
    
    # Optimize
    if time.time() - start_time < deadline - 10:
        # Use a penalty-based approach: minimize side length + penalty for overlaps
        def penalized_objective(params):
            if time.time() - start_time > deadline:
                return 1e10
            
            inner_hexes = []
            for i in range(N):
                x = params[3*i]
                y = params[3*i+1]
                a = params[3*i+2]
                inner_hexes.append((x, y, 1.0, a))
            
            # Compute side length
            side, success = compute_objective(params)
            if not success:
                return 1e10
            
            # Compute overlap penalty
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    # Approximate overlap by center distance
                    dx = inner_hexes[i][0] - inner_hexes[j][0]
                    dy = inner_hexes[i][1] - inner_hexes[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = 2.0  # minimum center distance for non-overlapping unit hexagons
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            return side + 10.0 * penalty
        
        # Try Nelder-Mead
        try:
            result = minimize(penalized_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True, 'maxfev': 2000})
            
            if result.fun < best_side + 10.0:
                # Check if the solution is feasible
                side, success = compute_objective(result.x)
                if success and side < best_side:
                    best_params = result.x
                    best_side = side
        except Exception as e:
            pass
        
        # Try a few random restarts
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() - start_time > deadline - 15:
                break
            
            # Random perturbation of initial config
            random_params = initial_params.copy()
            random_params += np.random.randn(len(random_params)) * 0.3
            # Random angles
            for i in range(N):
                random_params[3*i+2] = np.random.uniform(0, 60)
            
            try:
                result = minimize(penalized_objective, random_params, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8,
                                        'adaptive': True, 'maxfev': 1000})
                
                if result.fun < best_side + 10.0:
                    side, success = compute_objective(result.x)
                    if success and side < best_side:
                        best_params = result.x
                        best_side = side
            except Exception as e:
                pass
    
    # Extract final configuration
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    # Outer hexagon center and parameters
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    
    # Compute the actual minimum outer hexagon side length and optimal rotation
    best_outer_side = float('inf')
    best_outer_angle = 0.0
    
    for oangle_deg in range(0, 60, 2):
        oangle = math.radians(oangle_deg)
        
        max_support = 0.0
        for ih in inner:
            iv = hex_vertices(ih[0], ih[1], 1.0, ih[2])
            for vx, vy in iv:
                dx = vx - cx
                dy = vy - cy
                cos_a = math.cos(-oangle)
                sin_a = math.sin(-oangle)
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                
                for k in range(6):
                    normal_angle = math.radians(30 + 60 * k)
                    nx = math.cos(normal_angle)
                    ny = math.sin(normal_angle)
                    support = lx * nx + ly * ny
                    required_s = support / (math.sqrt(3) / 2.0)
                    if required_s > max_support:
                        max_support = required_s
        
        if max_support < best_outer_side:
            best_outer_side = max_support
            best_outer_angle = oangle_deg
    
    # Add small margin for numerical safety
    best_outer_side *= 1.001
    
    return inner, (cx, cy), best_outer_side, best_outer_angle

# EVOLVE-BLOCK-END
