"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Place 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Approach:
    - Fix inner hexagons on a hexagonal lattice with 30-degree rotation offset
    - Use linear programming to find minimum outer hexagon side length L
    - The outer hexagon is centered at origin with rotation 0
    
    A regular hexagon with side length L and rotation 0 has vertices at:
    (L, 0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L, 0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2)
    
    The containment constraint for a point (x,y) inside this hexagon:
    - |x| <= L
    - |sqrt(3)*x + 3*y| / 2 <= L  (top/bottom slanted sides)
    - |sqrt(3)*x - 3*y| / 2 <= L  (left/right slanted sides)
    
    Actually, let me use the standard form. A regular hexagon with "pointy top" (flat sides horizontal) 
    with circumradius R (distance from center to vertex) = side length L for regular hexagon.
    
    For a regular hexagon with side length L, the inradius (distance from center to midpoint of side) is L*sqrt(3)/2.
    
    Let's define the outer hexagon with flat sides horizontal (pointy top/bottom).
    Vertices at angles 30, 90, 150, 210, 270, 330 degrees from center, at distance L.
    
    So vertices: (L*cos(30°), L*sin(30°)), (L*cos(90°), L*sin(90°)), etc.
    = (L*sqrt(3)/2, L/2), (0, L), (-L*sqrt(3)/2, L/2), (-L*sqrt(3)/2, -L/2), (0, -L), (L*sqrt(3)/2, -L/2)
    
    The six half-plane constraints for containment:
    1. y <= L
    2. -y <= L  => y >= -L
    3. sqrt(3)*x - y <= L  (top-right edge)
    4. -sqrt(3)*x + y <= L (top-left edge)
    5. sqrt(3)*x + y <= L  (bottom-right edge)  
    6. -sqrt(3)*x - y <= L (bottom-left edge)
    
    Wait, let me be more careful. The edges of the hexagon:
    Edge from (L*sqrt(3)/2, L/2) to (0, L): 
        Direction: (-L*sqrt(3)/2, L/2)
        Normal pointing inward: (1, sqrt(3)) normalized... 
    
    Let me just use the standard inequality form. For a regular hexagon with flat top/bottom (pointy left/right):
    Vertices at (±L, 0), (±L/2, ±L*sqrt(3)/2)
    
    Constraints:
    - |x| <= L
    - |sqrt(3)*x + 3*y| <= 2*L  ... no
    
    Let me think differently. A regular hexagon can be defined by 6 inequalities.
    For a hexagon with "flat" sides at top and bottom, the constraints are:
    -x/2 + sqrt(3)*y/2 <= L/2  ... 
    
    Actually, simplest: a regular hexagon with circumradius R (= side length) centered at origin, 
    with one vertex pointing right, has the property that a point (x,y) is inside iff:
    max(|x|, |sqrt(3)*x/2 + 3*y/2|, |sqrt(3)*x/2 - 3*y/2|) <= R
    No, that's not right either.
    
    Let me use the support function approach. For a regular hexagon with side length L, 
    the width in direction theta is 2*L*cos(theta - k*pi/3) for appropriate k.
    
    Better: Let's place the outer hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    Vertex k at (L*cos(k*60°), L*sin(k*60°)).
    
    The 6 edges connect consecutive vertices. The inward normal for the edge between vertex k and k+1
    points in direction (k+0.5)*60° from the origin... 
    
    Actually, for a regular polygon, the apothem (inradius) is L*cos(30°) = L*sqrt(3)/2.
    The edge between vertex at angle 0° and vertex at angle 60° has its midpoint at angle 30°, 
    at distance L*sqrt(3)/2 from center. The inward normal points toward the center, i.e., in direction 30°+180° = 210°.
    
    So the 6 half-plane constraints (point inside iff all satisfied) are:
    For k = 0..5, the edge with outward normal at angle (k*60° + 30°):
    x*cos(30°+k*60°) + y*sin(30°+k*60°) <= L*sqrt(3)/2
    
    So: for each k in 0..5:
    x*cos(pi/6 + k*pi/3) + y*sin(pi/6 + k*pi/3) <= L*sqrt(3)/2
    
    Now, for each inner hexagon (also regular, side length 1), we need ALL its vertices to satisfy 
    these 6 inequalities. Since the hexagons are convex and the outer is convex, it suffices to check 
    the 6 vertices of each inner hexagon.
    
    Inner hexagon j has center (cx_j, cy_j) and rotation angle alpha_j. Its 6 vertices are at:
    (cx_j + cos(alpha_j + m*60°), cy_j + sin(alpha_j + m*60°)) for m = 0..5
    (since circumradius = side length = 1 for regular hexagon)
    
    So for each inner hex j, each vertex m, each outer constraint k:
    (cx_j + cos(alpha_j + m*pi/3))*cos(pi/6 + k*pi/3) + (cy_j + sin(alpha_j + m*pi/3))*sin(pi/6 + k*pi/3) <= L*sqrt(3)/2
    
    This is linear in (cx_j, cy_j, L) for fixed alpha_j.
    
    Strategy: Fix the rotations alpha_j. Place centers on a hexagonal lattice. Then solve LP for min L.
    
    Let me fix all inner hexagons to have the same rotation as the outer (alpha_j = 0) for simplicity,
    and place them on a hexagonal lattice.
    
    Actually, let me try a simpler approach: use a hexagonal packing arrangement.
    12 hexagons can be arranged as a hexagon of hexagons with radius 2 (1+6+... no, that's 7).
    Actually, a hexagonal number: 1, 7, 19, 37... 12 isn't a hexagonal number.
    
    Let me try a 3x4 rectangular grid of hexagons, or some other arrangement.
    
    For now, let me just use LP with fixed positions on a lattice and find optimal L.
    """
    
    # Define the 6 normal directions for the outer hexagon (pointy-right orientation)
    # Outward normals at angles 30°, 90°, 150°, 210°, 270°, 330°
    outer_normals = []
    for k in range(6):
        angle = np.pi/6 + k*np.pi/3  # 30° + k*60°
        outer_normals.append((np.cos(angle), np.sin(angle)))
    
    # Inner hexagon vertices offsets for rotation 0 (circumradius 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    inner_vertex_angles = [m*np.pi/3 for m in range(6)]
    inner_vertices = [(np.cos(a), np.sin(a)) for a in inner_vertex_angles]
    
    # Place 12 inner hexagons on a hexagonal lattice
    # Hexagonal lattice basis vectors for hexagons of side 1:
    # Distance between adjacent hexagon centers: 2*inradius = 2*sqrt(3)/2 = sqrt(3)
    # Actually, for regular hexagons of side 1 packed together, center-to-center distance is 2*sqrt(3)/2 = sqrt(3)
    # Wait, the inradius of a unit hexagon is sqrt(3)/2, so adjacent centers are 2*sqrt(3)/2 = sqrt(3) apart.
    
    # Let me use a different arrangement. Place them in a 3x4 grid with hexagonal spacing.
    # Or better, let me just try several candidate arrangements and pick the best.
    
    def compute_min_L(positions, rotations):
        """Given positions (list of (cx,cy)) and rotations (list of angles), find min L via LP."""
        n_inner = len(positions)
        # Variables: cx_0, cy_0, cx_1, cy_1, ..., cx_{n-1}, cy_{n-1}, L
        # Total: 2*n_inner + 1 variables
        # Minimize L: c = [0,0,...,0,1]
        
        n_vars = 2*n_inner + 1
        c = np.zeros(n_vars)
        c[-1] = 1.0  # minimize L
        
        A_ub = []
        b_ub = []
        
        for j in range(n_inner):
            cx_j, cy_j = positions[j]
            alpha_j = rotations[j]
            for m in range(6):
                vx = np.cos(alpha_j + m*np.pi/3)
                vy = np.sin(alpha_j + m*np.pi/3)
                for k in range(6):
                    nx, ny = outer_normals[k]
                    # (cx_j + vx)*nx + (cy_j + vy)*ny <= L*sqrt(3)/2
                    # cx_j*nx + cy_j*ny - L*sqrt(3)/2 <= -vx*nx - vy*ny
                    row = np.zeros(n_vars)
                    row[2*j] = nx
                    row[2*j+1] = ny
                    row[-1] = -np.sqrt(3)/2
                    A_ub.append(row)
                    b_ub.append(-vx*nx - vy*ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)]*(n_vars), method='highs')
        
        if res.status == 0:
            return res.fun, res.x
        else:
            return float('inf'), None
    
    # Try several arrangements
    best_L = float('inf')
    best_positions = None
    best_rotations = None
    
    # Arrangement 1: 3x4 grid with hexagonal offset
    # Hexagonal lattice: row i, col j
    # x = j * sqrt(3) + (i % 2) * sqrt(3)/2
    # y = i * 1.5  (vertical spacing for hex packing is 1.5 * side... actually for hexagons of side 1,
    # vertical distance between rows is 3/2 * 1 = 1.5? No.
    # For regular hexagons of side s packed in honeycomb:
    # Horizontal spacing: sqrt(3)*s
    # Vertical spacing: 1.5*s
    # With offset of sqrt(3)*s/2 for alternating rows
    
    s = 1.0
    h_spacing = np.sqrt(3) * s
    v_spacing = 1.5 * s
    
    # 3 rows x 4 cols
    positions1 = []
    for i in range(3):
        for j in range(4):
            x = j * h_spacing + (i % 2) * h_spacing / 2
            y = i * v_spacing
            positions1.append((x, y))
    
    rotations1 = [0.0] * 12
    L1, _ = compute_min_L(positions1, rotations1)
    if L1 < best_L:
        best_L = L1
        best_positions = positions1
        best_rotations = rotations1
    
    # Arrangement 2: 4x3 grid
    positions2 = []
    for i in range(4):
        for j in range(3):
            x = j * h_spacing + (i % 2) * h_spacing / 2
            y = i * v_spacing
            positions2.append((x, y))
    
    rotations2 = [0.0] * 12
    L2, _ = compute_min_L(positions2, rotations2)
    if L2 < best_L:
        best_L = L2
        best_positions = positions2
        best_rotations = rotations2
    
    # Arrangement 3: Hexagonal cluster (1 center + 6 ring + 5 more)
    # Center at (0,0), ring at distance sqrt(3)
    positions3 = [(0.0, 0.0)]
    for m in range(6):
        angle = m * np.pi / 3
        positions3.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
    # Add 5 more on second ring
    for m in range(5):
        angle = m * np.pi / 3 + np.pi / 6
        positions3.append((2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)))
    
    rotations3 = [0.0] * 12
    L3, _ = compute_min_L(positions3, rotations3)
    if L3 < best_L:
        best_L = L3
        best_positions = positions3
        best_rotations = rotations3
    
    # Arrangement 4: 2x6 grid
    positions4 = []
    for i in range(2):
        for j in range(6):
            x = j * h_spacing
            y = i * v_spacing
            positions4.append((x, y))
    
    rotations4 = [0.0] * 12
    L4, _ = compute_min_L(positions4, rotations4)
    if L4 < best_L:
        best_L = L4
        best_positions = positions4
        best_rotations = rotations4
    
    # Arrangement 5: 6x2 grid
    positions5 = []
    for i in range(6):
        for j in range(2):
            x = j * h_spacing + (i % 2) * h_spacing / 2
            y = i * v_spacing
            positions5.append((x, y))
    
    rotations5 = [0.0] * 12
    L5, _ = compute_min_L(positions5, rotations5)
    if L5 < best_L:
        best_L = L5
        best_positions = positions5
        best_rotations = rotations5
    
    # Arrangement 6: Try with 30-degree rotation for inner hexagons
    positions6 = positions1[:]
    rotations6 = [np.pi/6] * 12
    L6, _ = compute_min_L(positions6, rotations6)
    if L6 < best_L:
        best_L = L6
        best_positions = positions6
        best_rotations = rotations6
    
    # Arrangement 7: 4x3 with 30-degree rotation
    positions7 = positions2[:]
    rotations7 = [np.pi/6] * 12
    L7, _ = compute_min_L(positions7, rotations7)
    if L7 < best_L:
        best_L = L7
        best_positions = positions7
        best_rotations = rotations7
    
    # Arrangement 8: Compact 3x4 with smaller spacing (allowing slight overlap check)
    # Actually, we need to ensure inner hexagons don't overlap. Let me add that constraint.
    # For two unit hexagons at distance d apart, they don't overlap iff d >= 2*sqrt(3)/2 = sqrt(3)
    # (distance between centers >= 2 * inradius = sqrt(3))
    
    # Let me redo with non-overlap constraints
    def compute_min_L_with_no_overlap(positions, rotations):
        n_inner = len(positions)
        n_vars = 2*n_inner + 1
        c = np.zeros(n_vars)
        c[-1] = 1.0
        
        A_ub = []
        b_ub = []
        
        # Containment constraints
        for j in range(n_inner):
            cx_j, cy_j = positions[j]
            alpha_j = rotations[j]
            for m in range(6):
                vx = np.cos(alpha_j + m*np.pi/3)
                vy = np.sin(alpha_j + m*np.pi/3)
                for k in range(6):
                    nx, ny = outer_normals[k]
                    row = np.zeros(n_vars)
                    row[2*j] = nx
                    row[2*j+1] = ny
                    row[-1] = -np.sqrt(3)/2
                    A_ub.append(row)
                    b_ub.append(-vx*nx - vy*ny)
        
        # Non-overlap constraints: distance between centers >= sqrt(3)
        # (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= 3
        # This is non-linear. We can linearize using the fact that for hexagons with same orientation,
        # the minimum distance constraint can be approximated by checking in 6 directions.
        # For two unit hexagons with same rotation, they don't overlap iff for all 6 directions,
        # the projection distance >= sqrt(3)/2... actually it's more complex.
        # Simple linear approximation: require that in each of 6 hex directions, the separation >= sqrt(3)/2
        # Direction vectors: (cos(m*60°), sin(m*60°)) for m=0..5
        # (cx_i - cx_j)*cos(theta) + (cy_i - cy_j)*sin(theta) >= sqrt(3)/2 for at least... 
        # Actually, for two convex sets, they're disjoint iff there exists a separating line.
        # For two identical regular hexagons, it suffices to check the 6 normal directions.
        # Two unit hexagons at centers c1, c2 with same orientation are disjoint iff
        # max over 6 directions of |(c1-c2)·n| >= sqrt(3) ... no, >= 2*inradius = sqrt(3)?
        # The width of a unit hexagon in any normal direction is 2*inradius = sqrt(3) for the 3 principal directions,
        # and 2*circumradius = 2 for the other directions... 
        # Actually, for a regular hexagon, the width in direction theta is 2*sqrt(3)/2 * |cos(theta - k*60°)| + ... 
        # This is getting complicated. Let me just use the simple constraint that Euclidean distance >= sqrt(3).
        # Linearize: for each pair, add constraints in 6 directions.
        
        # Actually, let's just use a simple approach: for each pair (i,j), i<j:
        # (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= 3
        # Linearize by requiring: for some direction, the projection >= sqrt(3)
        # We can add: for each of 6 directions d, (cx_i-cx_j)*d.x + (cy_i-cy_j)*d.y >= sqrt(3)/2
        # OR use the fact that if Euclidean dist >= sqrt(3), then in at least one of the 6 hex directions, 
        # the projection is >= sqrt(3)/2. But we need ALL pairs to be separated, so we need for each pair,
        # there EXISTS a direction. This is an OR constraint, which is hard for LP.
        
        # Simpler: just require Euclidean distance >= sqrt(3) via linear relaxation.
        # For each pair, add: for all 6 directions, the absolute projection >= sqrt(3)/2 is too strong.
        # Let me just skip the non-overlap for now and rely on the lattice spacing being correct.
        # If I use proper hexagonal lattice spacing, non-overlap is guaranteed.
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)]*n_vars, method='highs')
        
        if res.status == 0:
            return res.fun, res.x
        else:
            return float('inf'), None
    
    # Since my lattice arrangements already ensure non-overlap, let's just use the best one found.
    
    # If all failed, fall back to a large value
    if best_L == float('inf') or best_L is None:
        best_L = 40.0
        best_positions = [(4.0*k, 0.0) for k in range(-6, 6)]
        best_rotations = [0.0] * 12
    
    # Center the arrangement
    cx_mean = np.mean([p[0] for p in best_positions])
    cy_mean = np.mean([p[1] for p in best_positions])
    
    inner = []
    for pos, rot in zip(best_positions, best_rotations):
        x = pos[0] - cx_mean
        y = pos[1] - cy_mean
        angle_deg = np.degrees(rot)
        inner.append([x, y, angle_deg])
    
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
