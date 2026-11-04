"""Pack 12 unit hexagons into minimal enclosing hexagon."""
import numpy as np
import time

N = 12

def construct():
    t0 = time.time()
    
    # Define a honeycomb-like arrangement for 12 hexagons
    # Rows of hexagons with alternating offsets
    # Using axial coordinates for hex grid
    # For 12 hexagons, try a compact arrangement
    
    # Hexagon vertices for unit hexagon at origin with rotation 0
    # Pointy-top: vertices at angles 30, 90, 150, 210, 270, 330 degrees
    # Flat-top: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    
    # Let's use flat-top hexagons (default orientation)
    # Vertices of unit flat-top hexagon centered at origin:
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    
    # In a hex grid with flat-top hexagons:
    # Horizontal spacing between centers: 1.5 (for adjacent columns)
    # Vertical spacing: sqrt(3) (for same column)
    # Odd columns shifted down by sqrt(3)/2
    
    # Let's place 12 hexagons in a compact pattern
    # Pattern: 3 columns of 4, or 4 columns of 3, etc.
    
    # Try: 4 columns x 3 rows with hex grid offset
    # Column 0: rows -1, 0, 1 -> x=0, y = -sqrt(3), 0, sqrt(3)
    # Column 1: rows -1, 0, 1 -> x=1.5, y = -sqrt(3)/2, sqrt(3)/2, 3*sqrt(3)/2 ... 
    
    # Actually, let me think more carefully.
    # For flat-top hexagons in a grid:
    # Center of hex at (col, row) in axial coords:
    # x = 1.5 * col
    # y = sqrt(3) * (row + col/2)
    
    # Let's use 4 columns, 3 rows each:
    # col 0,1,2,3; row -1,0,1
    
    sqrt3 = np.sqrt(3)
    centers = []
    for col in range(4):
        for row in range(-1, 2):  # -1, 0, 1
            x = 1.5 * col
            y = sqrt3 * (row + col / 2.0)
            centers.append((x, y))
    
    # That's 12 hexagons. Now compute the minimal enclosing regular hexagon.
    # The enclosing hexagon must contain all vertices of all inner hexagons.
    
    # Each inner hexagon has 6 vertices. For a hexagon at center (cx, cy) with rotation 0:
    # vertices at (cx + cos(k*pi/3), cy + sin(k*pi/3)) for k=0..5
    
    # Collect all vertices
    all_vertices = []
    for (cx, cy) in centers:
        for k in range(6):
            angle = k * np.pi / 3.0
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # Now find minimal enclosing regular hexagon.
    # A regular hexagon with side L, centered at (hx, hy), rotated by theta,
    # has vertices at (hx + L*cos(theta + k*pi/3), hy + L*sin(theta + k*pi/3)) for k=0..5
    # The hexagon contains point p if for all 6 edge normals, the dot product condition holds.
    
    # For a regular hexagon centered at origin with side L and rotation theta,
    # the support function in direction phi is L * cos(phi - theta - pi/6) / cos(pi/6) ... 
    # Actually, for a regular hexagon, the distance from center to each edge (apothem) is L * sqrt(3)/2.
    # The hexagon is the intersection of 6 half-planes.
    # Edge i is perpendicular to direction (theta + i*pi/3), at distance L*sqrt(3)/2 from center.
    
    # So for a hexagon with center (hx,hy), side L, rotation theta:
    # A point (x,y) is inside if for all i=0..5:
    #   (x-hx)*cos(theta + i*pi/3) + (y-hy)*sin(theta + i*pi/3) <= L * sqrt(3)/2
    
    # We need to find L, hx, hy, theta such that all vertices satisfy this.
    
    # For a given theta, the minimal L and center can be found by:
    # For each direction d_i = (cos(theta+i*pi/3), sin(theta+i*pi/3)),
    # the max projection of all vertices onto d_i gives the required extent.
    # L * sqrt(3)/2 >= max_i (max_v (v - c) . d_i) ... but we also need min.
    
    # Actually: for direction d, we need:
    #   max_v (v . d) - c . d <= L*sqrt(3)/2
    #   c . d - min_v (v . d) <= L*sqrt(3)/2
    # So: L*sqrt(3)/2 >= max(max_v(v.d) - c.d, c.d - min_v(v.d))
    # To minimize L for fixed theta, we want to choose c to balance these.
    # For each direction d_i, let m_i = max_v(v.d_i), n_i = min_v(v.d_i)
    # We need c.d_i such that max(m_i - c.d_i, c.d_i - n_i) is minimized.
    # Optimal c.d_i = (m_i + n_i)/2, giving extent (m_i - n_i)/2.
    # So L * sqrt(3)/2 >= max_i (m_i - n_i)/2
    # L >= max_i (m_i - n_i) / sqrt(3)
    
    # And the center c should satisfy c.d_i = (m_i + n_i)/2 for all i.
    # This is a linear system. For a hexagon, the 6 directions are not independent (3 pairs of opposites).
    # d_i and d_{i+3} are opposite: d_{i+3} = -d_i.
    # So m_{i+3} = max_v(v.(-d_i)) = -min_v(v.d_i) = -n_i
    # n_{i+3} = min_v(v.(-d_i)) = -max_v(v.d_i) = -m_i
    # So (m_{i+3} + n_{i+3})/2 = (-n_i - m_i)/2 = -(m_i + n_i)/2
    # And (m_{i+3} - n_{i+3})/2 = (-n_i + m_i)/2 = (m_i - n_i)/2. Good, consistent.
    
    # So we only need 3 independent directions. Let's use i=0,1,2.
    # Solve for c from:
    #   c . d_0 = (m_0 + n_0)/2
    #   c . d_1 = (m_1 + n_1)/2
    #   c . d_2 = (m_2 + n_2)/2
    
    # d_0, d_1, d_2 are 3 vectors in 2D. They're linearly dependent, so the system is overdetermined.
    # But if the points are contained in a hexagon, it should be consistent (or close).
    # Use least squares.
    
    best_L = float('inf')
    best_hx = best_hy = 0.0
    best_theta = 0.0
    
    # Try various rotations
    for theta in np.linspace(0, np.pi/3, 37):  # 37 angles in [0, 60 degrees]
        d = np.zeros((3, 2))
        for i in range(3):
            d[i, 0] = np.cos(theta + i * np.pi / 3)
            d[i, 1] = np.sin(theta + i * np.pi / 3)
        
        # Compute m_i and n_i for all 6 directions, but we only need 3 (the other 3 are determined)
        m = np.zeros(6)
        n = np.zeros(6)
        for i in range(6):
            angle = theta + i * np.pi / 3
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            projections = all_vertices @ dir_vec
            m[i] = np.max(projections)
            n[i] = np.min(projections)
        
        # L must satisfy L * sqrt(3)/2 >= max_i (m_i - n_i)/2
        # So L >= max_i (m_i - n_i) / sqrt(3)
        L_candidate = np.max(m - n) / sqrt3
        
        # Find center: solve c . d_i = (m_i + n_i)/2 for i=0,1,2
        A = d  # 3x2
        b = (m[:3] + n[:3]) / 2.0
        
        # Least squares
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        hx, hy = c[0], c[1]
        
        # Verify: check that L_candidate actually works with this center
        # For each of 6 directions, check that the constraints are satisfied
        valid = True
        for i in range(6):
            angle = theta + i * np.pi / 3
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            proj_center = hx * dir_vec[0] + hy * dir_vec[1]
            max_proj = np.max(all_vertices @ dir_vec)
            min_proj = np.min(all_vertices @ dir_vec)
            if max_proj - proj_center > L_candidate * sqrt3 / 2 + 1e-9:
                valid = False
                break
            if proj_center - min_proj > L_candidate * sqrt3 / 2 + 1e-9:
                valid = False
                break
        
        if valid and L_candidate < best_L:
            best_L = L_candidate
            best_hx = hx
            best_hy = hy
            best_theta = theta
    
    # Check time
    elapsed = time.time() - t0
    if elapsed > 120:
        # Return best found so far
        pass
    
    # Build inner list
    inner = []
    for (cx, cy) in centers:
        inner.append([float(cx), float(cy), 0.0])
    
    return inner, (float(best_hx), float(best_hy)), float(best_L), float(np.degrees(best_theta))

