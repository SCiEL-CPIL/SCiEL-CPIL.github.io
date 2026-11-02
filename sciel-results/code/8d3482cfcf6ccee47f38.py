"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing outer side length.
    Returns (inner, center, side, angle) where:
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (cx, cy) of outer hexagon
      side: side length of outer hexagon
      angle: rotation angle of outer hexagon (degrees)
    """
    # For N unit hexagons, a good arrangement is a hexagonal packing.
    # The optimal arrangement for small N is known:
    # N=1: 1
    # N=2: 2
    # N=3: 3 (triangle)
    # N=4: 4
    # N=5: 5
    # N=6: 6 (hexagonal)
    # N=7: 7 (center + 6 around)
    # N=11: 11 (center + 6 + 4 more, or some other arrangement)
    
    # For N=11, let's try a known good arrangement.
    # One approach: place hexagons in a hexagonal grid pattern.
    # A hexagonal number arrangement: 1, 7, 19, ...
    # For 11, we can do: center (1), first ring (6), second ring partial (4) = 11
    
    # Let's use a more systematic approach with optimization.
    
    # Unit hexagon vertices (flat-top orientation, angle=0):
    def hex_vertices(cx, cy, angle_deg, s=1.0):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.pi * np.arange(6) / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def outer_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.pi * np.arange(6) / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    
    def point_in_hexagon(p, center, side, angle_deg):
        """Check if point p is inside a regular hexagon with given center, side, angle."""
        # Transform point to hexagon's local frame
        dx = p[0] - center[0]
        dy = p[1] - center[1]
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        lx = cos_a * dx - sin_a * dy
        ly = sin_a * dx + cos_a * dy
        
        # For a regular hexagon with side s centered at origin, flat-top:
        # The hexagon is the intersection of 6 half-planes
        # Distance from center to each side (apothem) = s * sqrt(3) / 2
        apothem = side * np.sqrt(3) / 2
        
        # The 6 normals to the sides (for flat-top hexagon, angle=0):
        # Sides are at angles 0, 60, 120, 180, 240, 300 degrees from center
        # Actually for a regular hexagon with vertices at angles 0,60,...,300:
        # The sides are perpendicular to directions 30, 90, 150, 210, 270, 330
        side_angles = np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(side_angles), np.sin(side_angles)])
        
        # Project local point onto each normal
        projections = np.array([lx * n[0] + ly * n[1] for n in normals])
        
        # Point is inside if all projections <= apothem
        return np.all(projections <= apothem + 1e-9)
    
    def hexagons_disjoint(v1, v2, tol=1e-7):
        """Check if two hexagons (as vertex arrays) are disjoint (no interior overlap)."""
        # For simplicity, check if any vertex of one is inside the other
        # This is not sufficient for all cases but works for convex shapes if we check both ways
        # Actually for convex polygons, they are disjoint iff no vertex of one is inside the other
        # AND they don't share edges in a way that causes overlap... 
        # A simpler check: compute the minimum distance between edges, or use shapely-like logic
        # Since we don't have shapely, let's use a vertex-in-polygon check for both directions
        # plus edge intersection check
        
        def point_in_convex_poly(p, poly):
            """Check if point p is inside convex polygon poly (array of vertices)."""
            n = len(poly)
            sign = None
            for i in range(n):
                a = poly[i]
                b = poly[(i+1) % n]
                cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if abs(cross) < 1e-9:
                    continue
                s = np.sign(cross)
                if sign is None:
                    sign = s
                elif s != sign:
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
            if abs(d1) < 1e-9 and np.dot(p3-p1, p4-p1) >= -1e-9 and np.dot(p3-p1, p4-p1) <= np.dot(p4-p1, p4-p1) + 1e-9:
                return True
            if abs(d2) < 1e-9 and np.dot(p3-p2, p4-p2) >= -1e-9 and np.dot(p3-p2, p4-p2) <= np.dot(p4-p2, p4-p2) + 1e-9:
                return True
            if abs(d3) < 1e-9 and np.dot(p1-p3, p2-p3) >= -1e-9 and np.dot(p1-p3, p2-p3) <= np.dot(p2-p3, p2-p3) + 1e-9:
                return True
            if abs(d4) < 1e-9 and np.dot(p1-p4, p2-p4) >= -1e-9 and np.dot(p1-p4, p2-p4) <= np.dot(p2-p4, p2-p4) + 1e-9:
                return True
            return False
        
        # Check if any vertex of v1 is inside v2 or any vertex of v2 is inside v1
        for i in range(6):
            if point_in_convex_poly(v1[i], v2):
                return False
            if point_in_convex_poly(v2[i], v1):
                return False
        
        # Check edge intersections
        for i in range(6):
            j1 = (i+1) % 6
            for j in range(6):
                k = (j+1) % 6
                if segments_intersect(v1[i], v1[j1], v2[j], v2[k]):
                    return False
        
        return True
    
    def objective(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., xn, yn, an, cx, cy, side, outer_angle]
        Returns outer side length if valid, or a large penalty if invalid.
        """
        n = N
        hexes = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            hexes.append((x, y, a))
        
        cx = params[3*n]
        cy = params[3*n+1]
        side = params[3*n+2]
        outer_angle = params[3*n+3]
        
        # Ensure side is positive
        if side <= 0:
            return 1000.0
        
        # Check all inner hexagons are inside outer hexagon
        outer_v = outer_hex_vertices(cx, cy, side, outer_angle)
        for (x, y, a) in hexes:
            v = hex_vertices(x, y, a)
            # Check all vertices of inner hex are inside outer hex
            for pt in v:
                if not point_in_hexagon(pt, (cx, cy), side, outer_angle):
                    return side + 10.0  # penalty
        
        # Check inner hexagons are disjoint
        for i in range(n):
            vi = hex_vertices(hexes[i][0], hexes[i][1], hexes[i][2])
            for j in range(i+1, n):
                vj = hex_vertices(hexes[j][0], hexes[j][1], hexes[j][2])
                if not hexagons_disjoint(vi, vj):
                    return side + 10.0  # penalty
        
        return side
    
    # Initial guess: place hexagons in a hexagonal pattern
    # For N=11: center at (0,0), 6 around at distance 2*sqrt(3)/2*... 
    # Unit hexagon width (flat to flat) = sqrt(3), point to point = 2
    # For flat-top hexagons, adjacent centers in hex grid:
    # dx = sqrt(3), dy = 0 for horizontal neighbors
    # dx = sqrt(3)/2, dy = 1.5 for diagonal neighbors
    
    sqrt3 = np.sqrt(3)
    initial_hexes = []
    # Center
    initial_hexes.append((0.0, 0.0, 0.0))
    # First ring: 6 hexagons
    for k in range(6):
        angle = np.radians(60 * k)
        x = sqrt3 * np.cos(angle)
        y = 1.5 * np.sin(angle) * (2/np.sqrt(3)) * (np.sqrt(3)/1.5)  # let me recalculate
        # For flat-top hexagons, the 6 neighbors are at:
        # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
        angles = [0, 60, 120, 180, 240, 300]
        dists = [sqrt3, sqrt3, sqrt3, sqrt3, sqrt3, sqrt3]
        # Actually the distance between centers of adjacent flat-top hexagons is 2 * apothem = sqrt(3)
        # But the direction matters. Let me just use the known neighbor offsets:
        offsets = [(sqrt3, 0), (sqrt3/2, 1.5), (-sqrt3/2, 1.5), (-sqrt3, 0), (-sqrt3/2, -1.5), (sqrt3/2, -1.5)]
        x, y = offsets[k]
        initial_hexes.append((x, y, 0.0))
    # Second ring partial: 4 more hexagons to make 11 total
    # Add 4 more from the second ring
    # Second ring offsets (distance ~2*sqrt(3) in some directions)
    second_ring = [(2*sqrt3, 0), (1.5*sqrt3, 3), (-1.5*sqrt3, 3), (-2*sqrt3, 0), (-1.5*sqrt3, -3), (1.5*sqrt3, -3),
                   (sqrt3, 3), (-sqrt3, 3), (-sqrt3, -3), (sqrt3, -3)]
    # Pick 4 that are farthest from center to spread them out
    # Actually let's just pick 4 from the second ring
    for k in [0, 2, 4, 6]:
        x, y = second_ring[k]
        initial_hexes.append((x, y, 0.0))
    
    # Build initial params
    init_params = []
    for (x, y, a) in initial_hexes:
        init_params.extend([x, y, a])
    
    # Outer hexagon: center at (0,0), estimate side length
    # The bounding hexagon needs to contain all these hexagons
    # For now, guess a side length
    max_r = max(np.sqrt(x**2 + y**2) for (x, y, a) in initial_hexes)
    init_side = max_r + 2  # rough estimate
    init_params.extend([0.0, 0.0, init_side, 0.0])
    
    # Bounds for optimization
    bounds = []
    for i in range(N):
        bounds.extend([(-10, 10), (-10, 10), (0, 60)])
    bounds.extend([(-10, 10), (-10, 10), (1, 30), (0, 60)])
    
    # Run differential evolution
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42,
        maxiter=100,
        tol=1e-6,
        popsize=15,
        mutation=(0.3, 1.2),
        recombination=0.8,
        polish=False
    )
    
    best_params = result.x
    best_side = result.fun
    
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([float(x), float(y), float(a)])
    
    cx = float(best_params[3*N])
    cy = float(best_params[3*N+1])
    side = float(best_params[3*N+2])
    outer_angle = float(best_params[3*N+3])
    
    return inner, (cx, cy), side, outer_angle
# EVOLVE-BLOCK-END
