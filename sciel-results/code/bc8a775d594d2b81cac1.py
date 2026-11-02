import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define a compact honeycomb cluster of 12 unit hexagons
    # Using axial coordinates for hex grid, then converting to Cartesian
    # A compact 12-hex cluster: center + 6 neighbors + 5 more
    # Let's use a 3-row arrangement: 4, 4, 4 but shifted for hex packing
    
    # Hex lattice: pointy-top orientation, unit side length
    # Distance between centers of adjacent unit hexagons = sqrt(3)
    d = np.sqrt(3)
    
    # Define 12 positions in a compact cluster
    # Row-based: try a 4-4-4 arrangement with hex offsets
    # Actually, let's use a known compact shape: 
    # Positions in axial coords (q, r) for 12 hexagons
    # A good compact 12: 
    # (0,0), (1,0), (0,1), (1,1), (2,0), (2,1), (3,0), (3,1), (0,-1), (1,-1), (2,-1), (3,-1)
    # That's 12 hexagons in a 4x3 rectangular-ish block
    
    axial = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, -1), (1, -1), (2, -1), (3, -1)
    ]
    
    # Convert axial to Cartesian (pointy-top)
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2) * r
    positions = []
    for q, r in axial:
        x = d * (q + r / 2.0)
        y = d * (np.sqrt(3) / 2.0) * r
        positions.append([x, y])
    
    positions = np.array(positions)
    
    # Center the cluster at origin
    positions -= positions.mean(axis=0)
    
    # Now we need to find the minimal enclosing regular hexagon
    # Parameterize by rotation angle theta and scale (side length s)
    # For a given theta and s, check if all positions are inside the hexagon
    
    def point_in_hexagon(px, py, center_x, center_y, side, theta):
        """Check if point (px,py) is inside regular hexagon with given params."""
        # Transform point to hexagon's local frame
        dx = px - center_x
        dy = py - center_y
        # Rotate by -theta
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        lx = dx * cos_t - dy * sin_t
        ly = dx * sin_t + dy * cos_t
        
        # For a regular hexagon with side s, flat-top orientation:
        # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
        # Distance from center to each edge (apothem) = s * sqrt(3) / 2
        # The hexagon can be defined as intersection of 6 half-planes
        # For flat-top hexagon, edges are at angles 30, 90, 150, 210, 270, 330 from center
        # Normal vectors point outward at these angles
        
        apothem = side * np.sqrt(3) / 2.0
        for i in range(6):
            angle = np.pi / 6.0 + i * np.pi / 3.0  # 30, 90, 150, ...
            nx = np.cos(angle)
            ny = np.sin(angle)
            # Point must satisfy: nx*lx + ny*ly <= apothem
            if nx * lx + ny * ly > apothem + 1e-10:
                return False
        return True
    
    def all_inside(theta, side):
        """Check if all hexagon centers are inside the outer hexagon."""
        for px, py in positions:
            if not point_in_hexagon(px, py, 0.0, 0.0, side, theta):
                return False
        return True
    
    def min_side_for_rotation(theta):
        """Binary search on side length for a given rotation."""
        lo, hi = 0.0, 50.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if all_inside(theta, mid):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Ternary search on rotation angle
    # The problem has periodicity of 60 degrees (pi/3), so search in [0, pi/3)
    theta_lo = 0.0
    theta_hi = np.pi / 3.0
    
    # Sample some angles to find a good region
    best_theta = 0.0
    best_side = min_side_for_rotation(0.0)
    
    # Coarse search
    for theta in np.linspace(0, np.pi / 3, 37):
        s = min_side_for_rotation(theta)
        if s < best_side:
            best_side = s
            best_theta = theta
    
    # Fine ternary search around the best
    for _ in range(30):
        m1 = best_theta - (theta_hi - theta_lo) / 3.0
        m2 = best_theta + (theta_hi - theta_lo) / 3.0
        # Actually, let's do proper ternary search in a window around best_theta
        window = 0.05
        lo = max(0, best_theta - window)
        hi = min(np.pi / 3, best_theta + window)
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = min_side_for_rotation(m1)
        f2 = min_side_for_rotation(m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
        best_theta = (lo + hi) / 2.0
        window = (hi - lo)
        if window < 1e-6:
            break
    
    best_side = min_side_for_rotation(best_theta)
    
    # Build the result
    inner = []
    for px, py in positions:
        inner.append([px, py, 0.0])  # angle_degrees = 0 for each inner hexagon
    
    return inner, (0.0, 0.0), best_side, best_theta * 180.0 / np.pi
# EVOLVE-BLOCK-END
