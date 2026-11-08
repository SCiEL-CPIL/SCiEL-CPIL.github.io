import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(center, side, angle_deg=0.0):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    side: side length
    angle_deg: rotation in degrees. 0 means flat-top (vertices at 0, 60, ... degrees relative to center? No, standard math: vertex at angle 0 is (side, 0) if pointy-top? 
    Let's define:
    A regular hexagon with side length 's' has circumradius R=s.
    If angle=0, let's have a vertex at (s, 0). This is 'pointy-top' relative to x-axis.
    Vertices: (s*cos(theta + i*pi/3), s*sin(theta + i*pi/3)) for i in 0..5.
    """
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3.0
    cx, cy = center
    return np.array([[cx + side * np.cos(a), cy + side * np.sin(a)] for a in angles])

def _point_in_hexagon(point, center, side, angle_deg=0.0):
    """
    Check if point is inside hexagon.
    Hexagon is intersection of 6 half-planes.
    Distance from center to each side (inradius) is r = side * sqrt(3)/2.
    The normal vectors to the sides point inwards.
    For a hexagon with vertex at angle theta, the side normals are at theta + pi/6 + i*pi/3.
    Wait, if vertex is at theta, the side between theta and theta+pi/3 is perpendicular to the bisector theta+pi/6.
    The inradius is R_in = side * sqrt(3)/2.
    Condition: dot(p - c, n_i) <= R_in for all 6 outward normals? 
    Let's use the standard form.
    Distance from center to edge is d = side * sqrt(3) / 2.
    The normals to the edges are at angles phi_i = angle_deg + pi/6 + i*pi/3.
    For point p to be inside:
    sum over i: max(0, dot(p-c, n_i) - d) should be 0.
    """
    cx, cy = center
    px, py = point
    dx = px - cx
    dy = py - cy
    
    theta = np.deg2rad(angle_deg)
    # Normals to the edges. 
    # If vertex is at angle theta, the edge midpoint is at theta + pi/6.
    # The normal pointing OUT from the center to the edge is at theta + pi/6.
    # Wait, let's verify.
    # Vertex at 0: (s, 0). Vertex at 60: (s/2, s*sqrt(3)/2).
    # Midpoint: (3s/4, s*sqrt(3)/4).
    # Vector from center to midpoint: (3/4, sqrt(3)/4).
    # Angle: atan2(sqrt(3)/4, 3/4) = atan2(sqrt(3)/3, 1) = 30 deg = pi/6.
    # So yes, outward normal is at theta + pi/6 + i*pi/3.
    
    r_in = side * np.sqrt(3) / 2.0
    
    min_dist = np.inf
    for i in range(6):
        phi = theta + np.pi/6 + i * np.pi / 3.0
        nx = np.cos(phi)
        ny = np.sin(phi)
        # Projection of vector (dx, dy) onto normal (nx, ny)
        proj = dx * nx + dy * ny
        # If proj > r_in, point is outside this half-plane.
        # The "violation" is max(0, proj - r_in).
        # For distance calculation, we want the distance to the boundary.
        # If inside, distance to this edge is r_in - proj.
        # Min distance to boundary is min over all edges of (r_in - proj) if inside.
        # If outside, it's complicated (distance to line segment).
        # But for packing, we assume we are close to feasible.
        # We can use a simpler metric for optimization: 
        # "Penalty" = sum of max(0, proj - r_in)^2.
        # But we want to MINIMIZE outer size, so we want to maximize the minimum clearance.
        
    # Actually, for the bisection, we just need to check if a configuration fits.
    # A configuration fits if:
    # 1. No inner hexagons overlap.
    # 2. All inner hexagons are inside the outer hexagon.
    
    return True # Placeholder, will implement check separately

def _min_distance_between_hexagons(c1, s1, a1, c2, s2, a2):
    """
    Calculate minimum distance between two hexagons.
    Returns 0 if they overlap or touch.
    """
    # This is computationally expensive to do exactly for arbitrary rotations.
    # Approximation: Check distance between centers vs sum of inradii? No.
    # Check if any vertex of 1 is inside 2, or vertex of 2 inside 1.
    # And check edge intersections.
    
    # For a robust and fast check in a search context, we can use the following:
    # Two convex polygons are disjoint if there is a separating axis.
    # Axes are the normals of the edges of both polygons.
    
    v1 = _hexagon_vertices(c1, s1, a1)
    v2 = _hexagon_vertices(c2, s2, a2)
    
    # Get edge normals
    def get_normals(vertices):
        normals = []
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i+1)%6]
            edge = p2 - p1
            # Normal
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            normals.append(n)
        return normals
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    
    min_dist = np.inf
    
    for n in n1 + n2:
        # Project all vertices of v1 and v2 onto axis n
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Gap between the two intervals [min1, max1] and [min2, max2]
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
            
        if gap < min_dist:
            min_dist = gap
            
    return min_dist

def _is_inside_outer(inner_center, inner_side, inner_angle, outer_center, outer_side, outer_angle):
    """
    Check if inner hexagon is completely inside outer hexagon.
    """
    # The inner hexagon is inside the outer one if all its vertices are inside the outer one.
    # This is sufficient for convex polygons.
    v_inner = _hexagon_vertices(inner_center, inner_side, inner_angle)
    
    # Check each vertex against outer hexagon
    for p in v_inner:
        if not _point_in_hexagon(p, outer_center, outer_side, outer_angle):
            return False
    return True

def _check_validity(inner, outer_center, outer_side, outer_angle):
    """
    Check if the entire configuration is valid.
    inner: list of (x, y, angle_deg)
    """
    # 1. Check inner-inner collisions
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            c1 = (inner[i][0], inner[i][1])
            a1 = inner[i][2]
            c2 = (inner[j][0], inner[j][1])
            a2 = inner[j][2]
            
            dist = _min_distance_between_hexagons(c1, 1.0, a1, c2, 1.0, a2)
            if dist < 1e-9: # Overlap or touch
                return False

    # 2. Check inner-outer containment
    for i in range(n):
        c = (inner[i][0], inner[i][1])
        a = inner[i][2]
        if not _is_inside_outer(c, 1.0, a, outer_center, outer_side, outer_angle):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Strategy:
    # The optimal packing for N=11 is likely the "flower" pattern:
    # 1 central hexagon, 6 surrounding it, and 4 more? 
    # Actually, a hexagon of side length S can contain a hexagonal grid of side length k.
    # Number of hexagons in a hexagonal grid of "radius" k (where k=1 is just 1 hex) is 1 + 6 + 12 + ...
    # k=1: 1
    # k=2: 1 + 6 = 7
    # k=3: 1 + 6 + 12 = 19
    # So 11 hexagons don't form a perfect hexagonal grid.
    
    # However, 11 is close to 7.
    # A common optimal packing for 11 is 1 center + 6 neighbors + 4 more placed in the "gaps" or extending the shape.
    # Or simply a linear chain? No, that's long.
    
    # Let's try to fit a "compact" shape.
    # The side length of the outer hexagon required to contain a hexagonal cluster of 'radius' r (in units of hex side length)
    # If we have a hexagonal cluster of side length m (in terms of number of hexes on a side), the outer hexagon side length is roughly m + 1?
    # Actually, if we pack unit hexagons in a hexagonal lattice, the distance from center to the furthest vertex of the cluster determines the size.
    
    # Let's assume the optimal configuration for N=11 is a 3x3 grid minus 2 corners? Or a flower with 4 extra?
    # Let's define a specific candidate configuration:
    # Center at (0,0).
    # 6 neighbors at distance sqrt(3) (inradius * 2? No, center-to-center distance for touching hexes is 2 * inradius = sqrt(3)).
    # Wait, circumradius is 1. Inradius is sqrt(3)/2.
    # Distance between centers of touching hexes = 2 * inradius = sqrt(3).
    
    # Let's generate coordinates for a "hexagonal" cluster of 11 hexes.
    # Using axial coordinates (q, r).
    # Center (0,0).
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1). (6 hexes)
    # Next ring: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1). (12 hexes)
    # We need 11 total. So 1 center + 6 neighbors + 4 from the next ring.
    # To be compact, we should pick 4 from the next ring that are close to each other or symmetric.
    # Symmetry suggests taking 4 that form a rectangle or square?
    # Or maybe taking all 6 neighbors and 4 from the ring that extend the "arms"?
    
    # Let's try a specific layout:
    # 1 center
    # 6 neighbors
    # 4 more: (2,0), (0,2), (-2,1), (-1,-1) ? No, let's try to maintain symmetry.
    # How about a 3-4-3 arrangement?
    # Row 0: 3 hexes
    # Row 1: 4 hexes
    # Row 2: 3 hexes
    # Total 10. Need 11.
    # Row 0: 3
    # Row 1: 5
    # Row 2: 3
    # Total 11.
    
    # Let's convert axial to cartesian.
    # Axial (q, r) to Cartesian (x, y) for pointy-top hexes (vertex at top):
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Wait, standard pointy-top:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    
    # Let's use flat-top for easier alignment with outer hexagon if outer is flat-top.
    # Flat-top axial (q, r):
    # x = 3/2 * q
    # y = sqrt(3) * (r + q/2)
    
    # Let's stick to pointy-top for the lattice, and rotate the outer hexagon if needed.
    
    # Generate 11 points in a compact shape.
    # Let's try the "cross" shape extended.
    # Center (0,0)
    # N, S, E, W, NE, NW, SE, SW? No, 6 neighbors.
    
    # Let's just use a simple heuristic:
    # Place 11 hexes in a 3x4 grid-like structure?
    # Actually, the most compact shape for 11 is likely the 7-hex flower plus 4 more attached to the sides.
    
    # Let's define the positions in the lattice.
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Add 4 more: (2,0), (0,2), (-2,1), (-1,-1) -> This breaks symmetry.
    # How about (2,0), (2,-1), (-2,1), (-2,0)?
    # Let's try to find a symmetric set of 4 from the second ring.
    # The second ring has 12 positions.
    # If we pick (2,0), (0,2), (-2,2)?? No.
    # Let's pick 4 that are 90 degrees apart? Hex grid is 60 degrees.
    
    # Alternative: Just use a linear chain? No, side length would be ~10.
    # A 2x6 rectangle?
    # Side length of outer hexagon for a 2x6 cluster:
    # Width ~ 6 * sqrt(3) + something.
    # Height ~ 2 * 1.5 + something.
    
    # Let's use an optimization approach for the specific topology.
    # We will fix the topology (which hexes touch which) and optimize the positions?
    # No, for hexagons, the positions are rigid if they are touching in a lattice.
    # The only freedom is the rotation of the whole cluster and the position.
    # And the rotation of individual hexes?
    # If they are packed in a lattice, they are all aligned.
    # So we just need to find the minimal outer hexagon side length S that can contain this cluster.
    
    # Cluster definition:
    # We will try a few candidate topologies for N=11.
    
    def generate_cluster(topo_id):
        pts = []
        if topo_id == 0:
            # 7 flower + 4
            # (0,0)
            # 6 neighbors
            # 4 from ring 2: (2,0), (1,1), (-1,2), (-2,1) -> This is a contiguous arc of 4.
            # Let's try to make it symmetric.
            # (2,0), (0,2), (-2,2) is not symmetric.
            # How about (2,0), (0,2), (-2,0), (0,-2)? 
            # (0,2) is in ring 2? Yes. (0,-2) is in ring 2? Yes.
            # (2,0) and (-2,0) are in ring 2? Yes.
            # So: Center, 6 neighbors, and (2,0), (0,2), (-2,0), (0,-2).
            # Total 1 + 6 + 4 = 11.
            # This is a "plus" shape extended.
            pts.append((0,0))
            for q, r in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                pts.append((q,r))
            for q, r in [(2,0), (0,2), (-2,0), (0,-2)]:
                pts.append((q,r))
        elif topo_id == 1:
            # 3-5-3 rows
            # Row r=0: q = -1, 0, 1
            # Row r=1: q = -2, -1, 0, 1, 2
            # Row r=2: q = -1, 0, 1
            # Wait, in axial coords, "rows" are constant r.
            # For pointy-top, y depends on r.
            # Let's check connectivity.
            # (0,0) touches (1,0), (0,1), (-1,1)...
            # This shape might not be convex or compact.
            pass
            
        # Convert axial to cartesian (pointy-top, size=1)
        cart = []
        for q, r in pts:
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            cart.append((x, y))
        return cart

    # We will test topo 0.
    cluster_axial = generate_cluster(0)
    
    # Calculate cartesian coordinates
    cluster_pts = []
    for q, r in cluster_axial:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        cluster_pts.append((x, y))
        
    # The cluster has a specific shape. We need to find the minimal outer hexagon side S.
    # The outer hexagon can be rotated.
    # We will iterate over outer rotation angles from 0 to 60 degrees.
    # For each angle, we calculate the minimal S required to contain the cluster.
    # Since the cluster is fixed relative to itself, we can just scale the outer hexagon.
    # Actually, we can fix the outer hexagon center to the cluster centroid.
    
    centroid = np.mean(cluster_pts, axis=0)
    # Translate cluster so centroid is at (0,0)
    cluster_pts = [(x - centroid[0], y - centroid[1]) for x, y in cluster_pts]
    
    best_S = np.inf
    best_angle = 0.0
    
    # Search for optimal outer rotation
    # The cluster has symmetry. The optimal outer rotation will likely align with the cluster's symmetry.
    # Let's sample angles.
    
    for angle_deg in np.linspace(0, 60, 61):
        # For this outer angle, find minimal S
        # The constraint is that all vertices of all inner hexes must be inside the outer hex.
        # This is equivalent to: For each inner hex vertex v, the "hexagonal norm" of v relative to the outer hex must be <= S / S_unit?
        # No, the outer hex has side S. Its inradius is S * sqrt(3)/2.
        # The condition for a point p to be inside a hexagon of side S (centered at 0, rotated by alpha) is:
        # max_i |dot(p, n_i)| <= S * sqrt(3)/2
        # where n_i are the normals to the edges of the unit hexagon rotated by alpha.
        
        # So for each inner hex vertex v, we calculate:
        # D(v) = max_i |dot(v, n_i)|
        # We need D(v) <= S * sqrt(3)/2 for all v.
        # So S >= 2/sqrt(3) * max_v D(v).
        
        # We need to calculate max_v D(v) for all vertices of all inner hexes.
        
        # Get normals for outer hex of side 1 (scaled later)
        # Normals for unit hex (side 1) rotated by angle_deg:
        theta = np.deg2rad(angle_deg)
        # Normals at theta + pi/6 + i*pi/3
        normals = []
        for i in range(6):
            phi = theta + np.pi/6 + i * np.pi / 3.0
            normals.append(np.array([np.cos(phi), np.sin(phi)]))
        
        max_D = 0.0
        
        for px, py in cluster_pts:
            # Get vertices of this unit hexagon (aligned with lattice, so angle 0)
            # Wait, the inner hexagons are aligned with the lattice.
            # The lattice is "pointy-top".
            # So inner hexes have angle 0 (vertex at 0 deg? No, pointy-top has vertex at 90 deg? 
            # My cartesian conversion: x = sqrt(3)(q+r/2), y=1.5r.
            # For (0,1), x=0, y=1.5.
            # For (1,0), x=sqrt(3), y=0.
            # Distance from (0,0) to (0,1) is 1.5.
            # Distance from (0,0) to (1,0) is sqrt(3) ~ 1.732.
            # This is correct for pointy-top lattice.
            # The hexagons themselves are pointy-top.
            # So their vertices are at angles 30, 90, 150, 210, 270, 330?
            # Let's check. Center (0,0).
            # Vertex at top: (0, 1). Angle 90.
            # Vertex at bottom-right: (sqrt(3)/2, -0.5)? No.
            # Side length 1.
            # Vertices of pointy-top hex centered at (0,0):
            # (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5).
            # These are at angles 90, 30, -30, -90, -150, 150.
            # So angle_deg for _hexagon_vertices should be 90?
            # My _hexagon_vertices uses vertex at angle 0.
            # So for pointy-top, angle_deg = 90.
            
            v_hex = _hexagon_vertices((px, py), 1.0, 90.0)
            
            for v in v_hex:
                # Calculate max projection onto normals
                proj = np.dot(v, normals)
                # We need max |proj|
                d = np.max(np.abs(proj))
                if d > max_D:
                    max_D = d
                    
        # Required S
        S = 2.0 / np.sqrt(3.0) * max_D
        
        if S < best_S:
            best_S = S
            best_angle = angle_deg
            
        if time.time() > deadline:
            break

    # Construct the final result
    # Inner hexagons:
    inner = []
    for px, py in cluster_pts:
        inner.append([px, py, 90.0]) # Pointy-top
        
    # Outer hexagon:
    # Center (0,0)
    # Side best_S
    # Angle best_angle
    # Note: The outer hexagon angle in the return value is the rotation.
    # My _hexagon_vertices uses vertex at angle 0.
    # If best_angle is the rotation of the normals, the vertex rotation is best_angle - 30?
    # Let's check.
    # Normals at theta + 30.
    # Vertices at theta + 90?
    # If normals are at 30, 90, 150... (theta=0)
    # Vertices are at 0, 60, 120...
    # So if normals are rotated by theta, vertices are rotated by theta - 30?
    # No, if normal is at 30, vertex is at 0.
    # If normal is at 30+theta, vertex is at theta.
    # So vertex angle = theta.
    # My loop used theta for normals.
    # So outer angle_deg = best_angle.
    
    return inner, (0.0, 0.0), best_S, best_angle

# EVOLVE-BLOCK-END
