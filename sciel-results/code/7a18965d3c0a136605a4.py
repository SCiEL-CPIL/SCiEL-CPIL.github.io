import math
import time
import numpy as np

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    Vertex 0 is at angle 'angle_deg'.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _get_lattice_positions(n):
    """
    Generate positions for n unit hexagons in a hexagonal lattice.
    The center hexagon is at (0,0).
    Neighbors are at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300.
    This is the densest packing for regular hexagons.
    """
    positions = [(0.0, 0.0)]
    if n == 1:
        return positions
    
    # Hex grid coordinates (axial)
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # ...
    
    # We need to generate the first N positions in a compact hexagonal cluster.
    # The "radius" r in hex grid corresponds to 1 + 3r(r+1) hexagons.
    # r=0: 1
    # r=1: 7
    # r=2: 19
    # So for N=11, we take r=1 (7 hexes) + 4 from r=2.
    
    # Let's generate all positions up to radius 2 and sort by distance from center.
    candidates = []
    # Axial to Cartesian:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # For unit hexagons (side 1), distance between centers is sqrt(3).
    # So size = 1.
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Let's just iterate over a square range of axial coordinates
    # and filter by hex distance <= 2.
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Hex distance from (0,0)
            dist = (abs(q) + abs(r) + abs(-q-r)) / 2
            if dist <= 2:
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                candidates.append((x, y, dist))
    
    # Sort by distance, then by angle to make it deterministic and symmetric-ish
    # Actually, for N=11, we want the 7 center ones and the 4 closest from ring 2.
    # Ring 2 has 12 hexagons. They are all at the same distance.
    # To minimize the bounding box, we should pick 4 that are symmetric?
    # Or just pick the first 4?
    # Let's try to pick 4 that form a symmetric shape.
    # Ring 2 angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330?
    # No, hex grid ring 2 vertices are at angles 0, 60, 120, 180, 240, 300 for the "corners"
    # and 30, 90, 150, 210, 270, 330 for the "edges"?
    # Let's check coordinates for dist=2.
    # (2,0) -> x=2*sqrt(3), y=0. Angle 0.
    # (2,-1) -> x=sqrt(3)*(2-0.5)=1.5*sqrt(3), y=-1.5. Angle -30?
    # (1,-2) -> x=sqrt(3)*(1-1)=0, y=-3. Angle -90.
    # (-1,-1) -> x=sqrt(3)*(-1-0.5)=-1.5*sqrt(3), y=-1.5. Angle -150.
    # (-2,0) -> x=-2*sqrt(3), y=0. Angle 180.
    # (-2,1) -> x=sqrt(3)*(-2+0.5)=-1.5*sqrt(3), y=1.5. Angle 150.
    # (-1,2) -> x=sqrt(3)*(-1+1)=0, y=3. Angle 90.
    # (1,1) -> x=sqrt(3)*(1+0.5)=1.5*sqrt(3), y=1.5. Angle 30.
    # (2,-2) -> x=sqrt(3)*(2-1)=sqrt(3), y=-3. Angle -60? Wait.
    # Let's just sort by distance and take the first N.
    # To be safe and symmetric, I will manually select the 4 from ring 2.
    # The 7 in ring 0 and 1 are:
    # (0,0)
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # These form a hexagon of radius 1.5 (apothem of inner cluster?).
    # The 12 in ring 2 are at distance 2*sqrt(3) from center? No.
    # Distance from center for ring 1 is sqrt(3).
    # Distance from center for ring 2 is 2*sqrt(3).
    # Wait, (2,0) is at x=2*sqrt(3). Distance is 2*sqrt(3).
    # (2,-1) is at x=1.5*sqrt(3), y=-1.5. Dist = sqrt(6.75 + 2.25) = 3.
    # So (2,0) is closer than (2,-1)?
    # Hex distance is 2 for both.
    # Euclidean distance varies.
    # (2,0): dist = 2*sqrt(3) ≈ 3.464
    # (1,-2): dist = 3.0
    # So the "corner" hexagons of the ring are further away than the "edge" hexagons?
    # No, (1,-2) is an edge?
    # Let's list ring 2 points sorted by Euclidean distance from (0,0):
    # (1,-2): 3.0
    # (2,-1): 3.0
    # (2,0): 3.464
    # (1,1): 3.0
    # (0,2): 3.0
    # (-1,2): 3.0
    # (-2,1): 3.0
    # (-2,0): 3.464
    # (-1,-1): 3.0
    # (0,-2): 3.0
    # (-2,-1): 3.0? No, (-2,-1) is dist 3.
    # (2,-2): dist sqrt(3+9)=sqrt(12)=3.464
    
    # So there are 6 points at distance 3.0 and 6 at distance 3.464.
    # We need 4 more. The 4 closest are at distance 3.0.
    # Which 4? To minimize bounding box, we should pick them symmetrically.
    # The 6 at dist 3.0 are at angles:
    # (1,-2): (0, -3) -> 270 deg
    # (2,-1): (1.5*sqrt(3), -1.5) -> -30 deg (330)
    # (1,1): (1.5*sqrt(3), 1.5) -> 30 deg
    # (0,2): (0, 3) -> 90 deg
    # (-1,2): (0, 3) -> 90 deg? No, (-1,2) is x=0, y=3.
    # Wait, (1,1) and (-1,2) are different?
    # (1,1): q=1, r=1. x=sqrt(3)*(1.5), y=1.5.
    # (-1,2): q=-1, r=2. x=sqrt(3)*(-1+1)=0, y=3.
    # So (0,2) is q=0, r=2. x=0, y=3.
    # So (0,2) and (-1,2) are the same point? No.
    # Hex grid: (q,r).
    # (0,2) -> x=0, y=3.
    # (-1,2) -> x=sqrt(3)*(-1+1)=0, y=3.
    # Yes, they are the same point. My axial coordinates must be unique.
    # In axial coords, (q,r) is unique.
    # Let's re-verify (1,1) and (-1,2).
    # (1,1): x=1.5*sqrt(3), y=1.5.
    # (-1,2): x=0, y=3.
    # These are different.
    
    # Let's just generate all, sort by (dist, angle), and pick.
    # But for N=11, the "optimal" shape might not be a perfect hexagon + 4 random.
    # However, the hexagonal lattice is the densest. Any deviation increases area.
    # So we stick to lattice.
    # We pick the 7 closest (dist <= sqrt(3)) and then the 4 closest from the rest.
    
    candidates.sort(key=lambda p: (p[2], p[0], p[1])) # Sort by hex dist, then x, then y.
    # This is not ideal for symmetry.
    # Let's explicitly construct the 11 points.
    # Center: (0,0)
    # Ring 1 (6 points):
    r1 = []
    for i in range(6):
        a = math.radians(60 * i)
        x = math.sqrt(3) * math.cos(a)
        y = math.sqrt(3) * math.sin(a)
        r1.append((x, y))
    
    # Ring 2 candidates (12 points). We want 4.
    # To be symmetric, let's pick angles 0, 90, 180, 270?
    # Or 30, 150, 210, 330?
    # Let's pick the ones that are "between" the ring 1 vertices to fill gaps?
    # Ring 1 vertices are at 0, 60, 120, 180, 240, 300.
    # Ring 2 "corners" are at 0, 60, ... (aligned with ring 1).
    # Ring 2 "edges" are at 30, 90, ... (between ring 1).
    # The "edge" ones are closer to center (dist 3.0 vs 3.464).
    # So we should pick from the "edge" ones.
    # The 6 "edge" ones are at angles 30, 90, 150, 210, 270, 330.
    # We need 4. Let's pick 90, 270, 30, 330? (Vertical and Horizontal-ish?)
    # Or 90, 270, 150, 210?
    # Let's try 90, 270, 30, 330.
    # Points:
    # 90: (0, 3)
    # 270: (0, -3)
    # 30: (1.5*sqrt(3), 1.5)
    # 330: (1.5*sqrt(3), -1.5)
    # This is asymmetric.
    
    # Let's try 90, 270, 150, 210.
    # 90: (0, 3)
    # 270: (0, -3)
    # 150: (-1.5*sqrt(3), 1.5)
    # 210: (-1.5*sqrt(3), -1.5)
    # This is also asymmetric.
    
    # What if we pick 90, 270, 30, 210?
    # 90: (0, 3)
    # 270: (0, -3)
    # 30: (1.5*sqrt(3), 1.5)
    # 210: (-1.5*sqrt(3), -1.5)
    # This has point symmetry.
    
    # Let's just pick 4 arbitrary ones from the 6 closest and see.
    # Actually, the optimal packing for 11 hexagons is known to be a hexagon of 7 + 4.
    # The shape doesn't matter much for the bounding box if we can rotate the outer hex.
    # Let's pick (0,3), (0,-3), (1.5*sqrt(3), 1.5), (-1.5*sqrt(3), -1.5).
    
    r2_picks = [
        (0.0, 3.0),
        (0.0, -3.0),
        (1.5 * math.sqrt(3), 1.5),
        (-1.5 * math.sqrt(3), -1.5)
    ]
    
    positions = [(0.0, 0.0)] + r1 + r2_picks
    return positions

def _min_enclosing_hex_side(points, angle_deg):
    """
    Given a set of points and a rotation angle for the outer hexagon,
    find the minimum side length s and center (cx, cy) such that all points
    are inside the regular hexagon of side s centered at (cx, cy) with rotation angle_deg.
    
    A regular hexagon of side s and rotation alpha can be defined by 6 half-planes.
    The normals are at angles alpha + 30 + i*60.
    The distance from center to each side (apothem) is s * sqrt(3) / 2.
    
    For a point p to be inside, its projection onto each normal n_i must be <= apothem + (center projection).
    Let n_i be the unit normal.
    (p - c) . n_i <= A  => p . n_i - c . n_i <= A  => c . n_i >= p . n_i - A
    Also (p - c) . (-n_i) <= A => c . n_i >= -p . n_i - A
    
    So for each normal direction n_i, we have:
    c . n_i >= max_p (p . n_i - A)
    c . n_i >= max_p (-p . n_i - A)
    
    Let M_i = max_p (p . n_i) and m_i = min_p (p . n_i).
    Then c . n_i >= M_i - A
    And c . n_i >= -m_i - A
    
    So c . n_i >= max(M_i, -m_i) - A
    
    This must hold for all 6 normals.
    This is a linear programming problem for c, but we can solve it analytically for s.
    Actually, we want to minimize s (which is proportional to A).
    
    The condition is that the intersection of the half-planes defined by c . n_i >= K_i(A) is non-empty.
    Where K_i(A) = max(M_i, -m_i) - A.
    
    Alternatively, we can iterate over possible centers? No.
    
    Let's use the property that the optimal center is the center of the minimum enclosing circle? No.
    
    Let's use a simple approach:
    The width of the point set in direction n is W_n = M_n - m_n.
    The hexagon has width 2A in direction n (distance between opposite sides).
    So we need 2A >= W_n for all 3 pairs of normals?
    No, because the center can shift.
    
    If we fix the center c, the required A is max_i (max_p | (p-c) . n_i |).
    We want to minimize A.
    This is the Chebyshev center problem for a hexagon.
    
    Since the hexagon is centrally symmetric, the optimal center c is such that the max distance to the sides is minimized.
    
    Let's approximate: The optimal center is close to the centroid.
    Let's calculate the centroid.
    Then we can refine?
    
    Actually, for a set of points, the minimum enclosing regular hexagon of fixed orientation can be found by:
    1. Calculate M_i and m_i for the 3 normal directions (6 vectors, but 3 pairs).
    2. The required "radius" in direction i is R_i = (M_i - m_i) / 2 + |c . n_i - (M_i+m_i)/2|.
    Wait, the projection of the points onto n_i is [m_i, M_i].
    The projection of the hexagon onto n_i is [c . n_i - A, c . n_i + A].
    We need [m_i, M_i] subset [c . n_i - A, c . n_i + A].
    So c . n_i - A <= m_i  => c . n_i <= m_i + A
    c . n_i + A >= M_i  => c . n_i >= M_i - A
    
    So M_i - A <= c . n_i <= m_i + A.
    This requires M_i - A <= m_i + A  => 2A >= M_i - m_i.
    So A >= (M_i - m_i) / 2.
    
    If we choose c such that c . n_i = (M_i + m_i) / 2 for all i, then A = (M_i - m_i) / 2 works.
    But we can't satisfy this for all 3 directions simultaneously unless the point set is centrally symmetric with respect to that point.
    
    So we need to find c that minimizes A subject to:
    c . n_i >= M_i - A
    c . n_i <= m_i + A
    
    This is equivalent to minimizing A such that the intersection of the 6 strips is non-empty.
    
    We can solve this by iterating A from lower bound up.
    Lower bound A_min = max_i (M_i - m_i) / 2.
    
    For a given A, the feasible region for c is the intersection of 6 strips.
    If it's non-empty, we found a valid c.
    
    We can find the minimal A by binary search.
    For a given A, we can check if the intersection is non-empty using linear programming or simple geometry.
    Since it's 2D, we can just check if the intersection of the 6 half-planes is non-empty.
    This can be done by checking if the convex polygon defined by these half-planes has positive area (or is non-empty).
    
    Let's implement a simple check for non-empty intersection of half-planes.
    Or, since N is small, we can just pick the centroid as c and calculate the required A.
    Then we can try to improve c by moving it to reduce the max violation.
    
    Given the time budget, I will use the centroid as the center and calculate the side length.
    This might not be optimal, but it's fast and robust.
    To improve, I can sweep the angle and pick the best one.
    
    """
    # Calculate centroid
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    
    a = math.radians(angle_deg)
    apothem_required = 0.0
    
    for i in range(3):
        # Normal angle
        theta_n = a + math.radians(30 + i * 60)
        nx = math.cos(theta_n)
        ny = math.sin(theta_n)
        
        # Project points
        projs = [p[0]*nx + p[1]*ny for p in points]
        M = max(projs)
        m = min(projs)
        
        # Project center
        c_proj = cx*nx + cy*ny
        
        # Required apothem
        # c_proj - A <= m => A >= c_proj - m
        # c_proj + A >= M => A >= M - c_proj
        req_A = max(c_proj - m, M - c_proj)
        if req_A > apothem_required:
            apothem_required = req_A
            
    # Side length s = apothem * 2 / sqrt(3)
    s = apothem_required * 2.0 / math.sqrt(3)
    return s, (cx, cy)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate inner positions
    positions = _get_lattice_positions(N)
    
    # Inner hexagons are all aligned (angle 0)
    inner = []
    for x, y in positions:
        inner.append([x, y, 0.0])
    
    # 2. Find best outer hexagon
    best_s = float('inf')
    best_c = (0, 0)
    best_angle = 0.0
    
    # Sweep angle from 0 to 60 degrees (periodicity)
    # Fine sweep
    for deg in np.arange(0, 60.01, 0.5):
        s, c = _min_enclosing_hex_side(positions, deg)
        if s < best_s:
            best_s = s
            best_c = c
            best_angle = deg
            
    # Refine around best angle
    if time.time() < deadline - 5:
        for deg in np.arange(best_angle - 2, best_angle + 2.01, 0.1):
            s, c = _min_enclosing_hex_side(positions, deg)
            if s < best_s:
                best_s = s
                best_c = c
                best_angle = deg
                
    return inner, best_c, best_s, best_angle

if __name__ == "__main__":
    r = construct()
    print(r)
