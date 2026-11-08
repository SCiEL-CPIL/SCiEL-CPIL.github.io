"""Fresh design: hexagonal close-packed arrangement of unit hexagons."""
import math
import time

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point is inside hexagon using half-plane tests."""
    a = math.radians(angle_deg)
    # For a regular hexagon, check distance to each edge
    for i in range(6):
        theta = a + math.radians(60 * i)
        # Edge normal points outward
        nx = math.cos(theta + math.radians(30))
        ny = math.sin(theta + math.radians(30))
        # Distance from center to edge (apothem)
        apothem = side * math.cos(math.radians(30))
        dx = px - cx
        dy = py - cy
        if dx * nx + dy * ny > apothem + 1e-9:
            return False
    return True

def _hexagons_disjoint(v1, v2, v3, v4, eps=1e-9):
    """Check if two hexagons (given as vertex lists) are disjoint (no overlap)."""
    # Check if any vertex of one is inside the other
    for p in v1:
        if _point_in_hexagon(p[0], p[1], v2[0][0], v2[0][1], 1.0, 0.0):
            # This is a rough check; need proper polygon intersection
            pass
    # Use separating axis theorem for convex polygons
    def get_axes(verts):
        axes = []
        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            dx = verts[j][0] - verts[i][0]
            dy = verts[j][1] - verts[i][1]
            # Normal
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-12:
                continue
            axes.append((-dy/length, dx/length))
        return axes
    
    def project(verts, axis):
        dots = [v[0]*axis[0] + v[1]*axis[1] for v in verts]
        return min(dots), max(dots)
    
    axes = get_axes(v1) + get_axes(v2)
    for ax in axes:
        min1, max1 = project(v1, ax)
        min2, max2 = project(v2, ax)
        if max1 < min2 - eps or max2 < min1 - eps:
            return True  # Disjoint
    return False  # Overlapping

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    N = 11  # Default from original
    
    # Try different hexagonal cluster sizes
    # For a hexagonal cluster with "radius" r, count = 3r(r+1)+1
    # r=0: 1, r=1: 7, r=2: 19, r=3: 31, etc.
    
    # For N hexagons, find the smallest hexagonal cluster that contains N
    # and compute the enclosing hexagon size
    
    # Hexagonal grid: centers at (col, row) in axial coordinates
    # Convert to Cartesian: x = sqrt(3)*(col + row/2), y = 1.5*row (for pointy-top)
    # Or for flat-top: x = 1.5*col, y = sqrt(3)*(row + col/2)
    
    # Let's use flat-top hexagons with side=1
    # Center spacing: horizontal = 1.5, vertical offset = sqrt(3)/2
    # Actually for flat-top: adjacent centers are sqrt(3) apart
    
    # Simpler: use the fact that in a hexagonal packing, the distance between
    # centers of adjacent hexagons is 2*apothem = sqrt(3)
    
    # For N hexagons arranged in a hexagonal cluster:
    # Generate positions in a hexagonal grid pattern
    
    def hex_cluster_positions(r):
        """Generate positions for a hexagonal cluster of radius r (flat-top hexagons)."""
        positions = []
        for q in range(-r, r+1):
            for s in range(max(-r, -q-r), min(r, -q+r)+1):
                t = -q - s
                # Convert axial (q,s) to Cartesian for flat-top hexagons
                # x = sqrt(3) * (q + s/2)
                # y = 3/2 * s
                x = math.sqrt(3) * (q + s/2)
                y = 1.5 * s
                positions.append((x, y))
        return positions
    
    def enclosing_hexagon_size(positions):
        """Compute the minimum side length of an enclosing regular hexagon."""
        if not positions:
            return 0.0
        
        # The enclosing hexagon must contain all unit hexagons
        # For a set of hexagon centers, the enclosing hexagon side length
        # can be computed by finding the maximum extent in each of the 6 directions
        
        # For a regular hexagon centered at origin with side L, the apothem is L*sqrt(3)/2
        # A point (x,y) is inside if for all 6 edge normals n_i: dot((x,y), n_i) <= apothem
        
        # We need to find L such that all unit hexagons fit
        # Each unit hexagon has extent 1 in all directions from its center
        
        # The effective radius of each unit hexagon in direction n is 1 (circumradius)
        # So we need: max over all centers and all 6 directions of (dot(center, n) + 1) <= L*sqrt(3)/2
        
        best_L = 0.0
        # Try different orientations for the enclosing hexagon
        for enc_angle in range(0, 30, 5):  # 0 to 25 degrees in steps of 5
            a = math.radians(enc_angle)
            max_extent = 0.0
            for i in range(6):
                theta = a + math.radians(60 * i)
                nx = math.cos(theta)
                ny = math.sin(theta)
                for (cx, cy) in positions:
                    extent = cx*nx + cy*ny + 1.0  # +1 for the unit hexagon's circumradius
                    if extent > max_extent:
                        max_extent = extent
            # apothem = L * sqrt(3)/2, so L = max_extent / (sqrt(3)/2)
            L = max_extent / (math.sqrt(3)/2)
            if L > best_L:
                best_L = L
        
        return best_L
    
    # Find best arrangement for N hexagons
    best_result = None
    best_side = float('inf')
    
    # Strategy: use a hexagonal cluster that can hold N hexagons
    # Find smallest r such that 3r(r+1)+1 >= N
    r_min = 0
    while 3*r_min*(r_min+1) + 1 < N:
        r_min += 1
    
    # Try r_min and r_min+1
    for r in range(r_min, r_min + 3):
        if time.time() > deadline:
            break
        
        positions = hex_cluster_positions(r)
        if len(positions) < N:
            continue
        
        # Take first N positions (or try different subsets)
        # For now, take the first N
        selected = positions[:N]
        
        side = enclosing_hexagon_size(selected)
        
        if side < best_side:
            best_side = side
            best_result = (selected, side)
    
    # If no valid result, use a simple linear arrangement
    if best_result is None:
        # Linear arrangement
        positions = [(math.sqrt(3)*i, 0.0) for i in range(N)]
        side = enclosing_hexagon_size(positions)
        best_result = (positions, side)
    
    selected, side = best_result
    
    # Build inner hexagons
    inner = []
    for (x, y) in selected:
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), side, 0.0
