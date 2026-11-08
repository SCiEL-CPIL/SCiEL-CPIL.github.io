"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Unit hexagon has inradius (apothem) = sqrt(3)/2, circumradius = 1
    # Centers of adjacent hexagons in perfect tiling are 2 * apothem = sqrt(3) apart
    
    # Place hexagons on a honeycomb lattice
    # Use axial coordinates for hex grid
    # For N=11, use a compact arrangement:
    # Center + 6 neighbors + 4 from second ring
    
    # Hex grid spacing: sqrt(3) between adjacent centers
    s = np.sqrt(3.0)
    
    # Axial coordinates (q, r) to pixel (x, y) for pointy-top hexagons
    # x = s * (q + r/2), y = 1.5 * r  (for unit hexagons with side 1)
    # Actually, let's use the standard: for hexagons of side length 1,
    # the distance between centers of adjacent hexagons is sqrt(3)
    
    # Define positions in axial coordinates
    # Center: (0,0)
    # First ring (6 hexagons): (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Second ring partial: pick 4 from (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,1)
    # Let's pick a symmetric subset
    
    axial_coords = [
        (0, 0),  # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # first ring
        (2, 0), (2, -1), (1, -2), (-1, -1),  # 4 from second ring
    ]
    
    # Convert axial to Cartesian
    # For pointy-top: x = sqrt(3) * (q + r/2), y = 3/2 * r
    # But we want the hexagons to tile, so let's verify
    # Actually, for regular hexagons of side 1, the center-to-center distance 
    # for edge-sharing is sqrt(3). The angle between adjacent centers is 60 degrees.
    
    # Let's use a simpler approach: place centers at hex grid positions
    # with spacing sqrt(3)
    
    inner = []
    for q, r in axial_coords:
        x = s * (q + r / 2.0)
        y = 1.5 * r
        angle = 0.0  # all same orientation
        inner.append([x, y, angle])
    
    # Compute minimal enclosing regular hexagon
    # A regular hexagon with center (cx, cy), side length S, rotation theta
    # has 6 sides. The constraint is that all inner hexagon vertices must be inside.
    
    # For a unit hexagon at position (px, py) with rotation phi, its vertices are at:
    # (px + cos(phi + k*pi/3), py + sin(phi + k*pi/3)) for k=0..5
    
    # The outer hexagon with center (cx,cy), side S, rotation theta has support function:
    # h(theta + k*pi/3) = S * sqrt(3)/2 for k=0..5 (the apothem is S*sqrt(3)/2)
    
    # Actually, for a regular hexagon of side S, the apothem (distance from center to side) is S*sqrt(3)/2
    # and the circumradius (center to vertex) is S.
    
    # The outer hexagon contains a point (x,y) if for all 6 directions n_k (unit normals to sides):
    # (x - cx) * n_k + (y - cy) * n_k <= S * sqrt(3)/2
    
    # To minimize S, we need S >= max over all points and all 6 directions of:
    # ((x - cx) * n_k + (y - cy) * n_k) / (sqrt(3)/2)
    
    # We can also optimize cx, cy, theta.
    
    # For simplicity, let's assume the outer hexagon is centered at the centroid of inner hexagons
    # and has the same orientation as the lattice (theta = 0)
    
    # Compute centroid
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for px, py, phi in inner:
        phi_rad = np.radians(phi)
        for k in range(6):
            angle = phi_rad + k * np.pi / 3
            vx = px + np.cos(angle)
            vy = py + np.sin(angle)
            all_vertices.append((vx, vy))
    
    # For outer hexagon with orientation theta, the 6 normal directions are at angles theta + k*pi/3
    # We need to find theta that minimizes the maximum projection
    
    # Let's try theta = 0 first
    theta = 0.0
    max_proj = 0
    for vx, vy in all_vertices:
        dx = vx - cx
        dy = vy - cy
        for k in range(6):
            angle = theta + k * np.pi / 3
            nx = np.cos(angle)
            ny = np.sin(angle)
            proj = dx * nx + dy * ny
            if proj > max_proj:
                max_proj = proj
    
    # S * sqrt(3)/2 >= max_proj, so S >= max_proj * 2/sqrt(3)
    S = max_proj * 2.0 / np.sqrt(3.0)
    
    # Try to optimize theta
    best_S = S
    best_theta = theta
    for theta_try in np.linspace(0, np.pi/3, 37):
        max_proj = 0
        for vx, vy in all_vertices:
            dx = vx - cx
            dy = vy - cy
            for k in range(6):
                angle = theta_try + k * np.pi / 3
                nx = np.cos(angle)
                ny = np.sin(angle)
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
        S_try = max_proj * 2.0 / np.sqrt(3.0)
        if S_try < best_S:
            best_S = S_try
            best_theta = theta_try
    
    return inner, (cx, cy), best_S, np.degrees(best_theta)
# EVOLVE-BLOCK-END
