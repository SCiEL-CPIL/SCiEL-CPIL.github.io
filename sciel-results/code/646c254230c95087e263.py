<<<<<<< SEARCH
def initial_cluster():
    """11 hexagons in a hexagonal arrangement (1 center + 2 rings), side 1."""
    # Axial coordinates for hex radius 2 (12 cells) minus one? We need 11.
    # Standard "hex of side 2" has 1+6+12=19. We want 11: use center + 6 + 4 of ring2? 
    # Simpler: center + 6 neighbors (ring1) + 4 extra placed in gaps.
    # Actually a compact 11-hex cluster: use the 11 cells of a "hexagon of side 2" 
    # is 19. Instead use a 3x3-ish block. Let's use axial coords within radius 1 
    # (7 cells) plus 4 more.
    # For a clean start, use 7 (radius-1 hex) + 4 = 11, placed to be compact.
    pts = []
    angs = []
    # radius-1 hexagon (7 cells)
    axial = [(0, 0)]
    for k in range(6):
        a = np.deg2rad(60 * k)
        axial.append((math.cos(a), math.sin(a)))  # unit axial -> cartesian approx
    # Convert axial (cube) to cartesian for pointy-top: x = (q + r/2)*sqrt(3), y = r*1.5
    # Use a proper axial set for radius 1:
    axial_r1 = [(0,0), (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    def axial_to_xy(q, r):
        return (math.sqrt(3) * (q + r / 2.0), 1.5 * r)
    for (q, r) in axial_r1:
        x, y = axial_to_xy(q, r)
        pts.append([x, y])
        angs.append(0.0)
    # 4 more cells: extend ring 2 partially to keep compact (pick 4 of the 12 ring-2 cells)
    # ring-2 axial coords (radius 2)
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                ring2.append((q, r))
    # pick 4 that are "aligned" to reduce bounding box: choose those with small |y|
    ring2.sort(key=lambda t: abs(1.5 * t[1]))
    for (q, r) in ring2[:4]:
        x, y = axial_to_xy(q, r)
        pts.append([x, y])
        angs.append(0.0)
    return pts, angs
=======
def initial_cluster():
    """11 hexagons in a compact cluster, correctly spaced."""
    pts = []
    angs = []
    
    def axial_to_xy(q, r):
        # Pointy-top hexagon grid spacing
        # Distance between centers of adjacent hexagons is sqrt(3) * side_length
        # Here side_length = 1, so distance = sqrt(3)
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)

    # We need 11 hexagons.
    # A hexagon of "radius" 1 (side length 1 in grid terms) has 7 hexagons (center + 6 neighbors).
    # We need 4 more.
    # The next ring (radius 2) has 12 hexagons.
    # Let's take the 7 from radius 1 and 4 from radius 2.
    # To keep it compact, we pick 4 from radius 2 that are "close" to the existing shape,
    # ideally filling in one side to make it more rectangular or square-like if possible,
    # or just the first 4 encountered.
    
    # Radius 1 axial coordinates
    r1_coords = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for q, r in r1_coords:
        x, y = axial_to_xy(q, r)
        pts.append([x, y])
        angs.append(0.0)
        
    # Radius 2 axial coordinates
    r2_coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                r2_coords.append((q, r))
                
    # Sort radius 2 coords to pick the "top" ones first (smaller y) to keep bounding box tight
    # Or just pick 4 arbitrary ones. Let's pick the ones with smallest y (top-most)
    # to try to make the cluster as compact as possible in one direction.
    r2_coords.sort(key=lambda t: axial_to_xy(t[0], t[1])[1])
    
    for q, r in r2_coords[:4]:
        x, y = axial_to_xy(q, r)
        pts.append([x, y])
        angs.append(0.0)
        
    return pts, angs
>>>>>>> REPLACE

