"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """
    Return the 6 vertices of a regular hexagon with side length 'side',
    centered at (cx, cy), rotated by angle_deg degrees.
    Vertices are ordered counter-clockwise.
    """
    angle_rad = math.radians(angle_deg)
    # For a regular hexagon with side length s, the distance from center to vertex is s.
    # The vertices are at angles 0, 60, 120, 180, 240, 300 degrees relative to the center,
    # plus the rotation offset.
    # However, standard orientation: flat top? Pointy top?
    # The problem says "regular hexagons". Usually side length s implies distance between parallel sides is s*sqrt(3).
    # Let's use the standard vertex definition:
    # v_k = center + side * (cos(angle + k*pi/3), sin(angle + k*pi/3))
    # This creates a hexagon where the distance from center to vertex is 'side'.
    # Wait, for a regular polygon, side length = 2 * R * sin(pi/n).
    # So R = side / (2 * sin(pi/6)) = side / (2 * 0.5) = side.
    # So yes, distance from center to vertex is equal to side length.
    
    angles = angle_rad + np.arange(6) * (math.pi / 3)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _polygon_area(pts):
    """
    Calculate area of a polygon given its vertices.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _polygon_contains_point(poly, point):
    """
    Check if a point is inside or on the boundary of a convex polygon.
    poly: (N, 2) array of vertices in CCW order.
    point: (2,) array.
    """
    # Use cross product to check if point is to the left of all edges (for CCW polygon)
    # If any cross product is negative, the point is outside.
    # Cross product of (v2 - v1) and (p - v1)
    # If all cross products are >= 0, point is inside.
    
    # For robustness, we can also check if the point is within the bounding box first.
    # But cross product is sufficient for convex polygons.
    
    n = len(poly)
    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i + 1) % n]
        edge = v2 - v1
        vec = point - v1
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        if cross < -1e-9: # Tolerance for floating point
            return False
    return True

def _polygons_disjoint(poly1, poly2, tol=1e-9):
    """
    Check if two convex polygons are disjoint (no overlap in interior).
    They can touch at edges or vertices.
    """
    # 1. Check if any vertex of poly1 is inside poly2
    for p in poly1:
        if _polygon_contains_point(poly2, p):
            return False
    # 2. Check if any vertex of poly2 is inside poly1
    for p in poly2:
        if _polygon_contains_point(poly1, p):
            return False
    # 3. Check for edge intersections
    # For convex polygons, if no vertices are inside and no edges cross, they are disjoint or touching.
    # We need to check if any edges intersect properly.
    n1 = len(poly1)
    n2 = len(poly2)
    
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
    def on_segment(A, B, C):
        # Check if B is on segment AC
        if min(A[0], C[0]) <= B[0] <= max(A[0], C[0]) and min(A[1], C[1]) <= B[1] <= max(A[1], C[1]):
            return True
        return False
        
    def intersect(p1, q1, p2, q2):
        # Check if segment p1-q1 intersects segment p2-q2
        d1 = ccw(p2, q2, p1)
        d2 = ccw(p2, q2, q1)
        d3 = ccw(p1, q1, p2)
        d4 = ccw(p1, q1, q2)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
            
        if d1 == 0 and on_segment(p2, p1, q2): return True
        if d2 == 0 and on_segment(p2, q1, q2): return True
        if d3 == 0 and on_segment(p1, p2, q1): return True
        if d4 == 0 and on_segment(p1, q2, q1): return True
        
        return False

    for i in range(n1):
        p1 = poly1[i]
        q1 = poly1[(i + 1) % n1]
        for j in range(n2):
            p2 = poly2[j]
            q2 = poly2[(j + 1) % n2]
            if intersect(p1, q1, p2, q2):
                return False
                
    return True

def _min_enclosing_hexagon_side(inner_polys, center, rotation_deg):
    """
    Calculate the minimum side length of a regular hexagon centered at 'center'
    with rotation 'rotation_deg' that contains all 'inner_polys'.
    If no such hexagon exists (which shouldn't happen if we scale up), return infinity.
    
    Actually, we want to find the side length S such that all inner hexagons are inside
    the outer hexagon.
    The outer hexagon is defined by center, S, and rotation_deg.
    The condition is: for all vertices v of all inner hexagons, v must be inside the outer hexagon.
    
    For a regular hexagon centered at origin with rotation 0, the boundaries are defined by:
    |x| <= S
    |x/2 + y*sqrt(3)/2| <= S * sqrt(3)/2  (Wait, let's derive properly)
    
    Let's use the support function approach.
    For a convex polygon P, the width in direction u is max_{p in P} dot(p, u).
    The outer hexagon H(S, rot) has width w(u) in direction u.
    For H to contain P, we need max_{p in P} dot(p - center, u) <= w_H(u) for all directions u.
    Actually, it's simpler to just check vertices.
    
    Let's transform the problem to the outer hexagon's local frame.
    Rotate all inner polygons by -rotation_deg around center.
    Then the outer hexagon is axis-aligned (rotation 0).
    For an axis-aligned regular hexagon with side length S centered at origin:
    The vertices are at (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2).
    
    The inequalities for a point (x, y) to be inside this hexagon are:
    1. |x| <= S
    2. |x/2 + y*sqrt(3)/2| <= S * sqrt(3)/2  -->  |x + y*sqrt(3)| <= S * sqrt(3)
    3. |-x/2 + y*sqrt(3)/2| <= S * sqrt(3)/2  -->  |-x + y*sqrt(3)| <= S * sqrt(3)
    
    Wait, let's verify.
    The normal vectors to the sides are at angles 0, 60, 120, 180, 240, 300.
    Side 1 (right): normal (1, 0). Distance from origin to side is S * cos(30) = S * sqrt(3)/2.
    So x <= S * sqrt(3)/2? No.
    Vertex is (S, 0). The side connecting (S,0) and (S/2, S*sqrt(3)/2).
    Midpoint is (3S/4, S*sqrt(3)/4).
    Normal vector is perpendicular to edge. Edge vector is (-S/2, S*sqrt(3)/2). Normal is (sqrt(3)/2, 1/2).
    Equation: sqrt(3)/2 * x + 1/2 * y = C.
    Plug in (S, 0): sqrt(3)/2 * S = C.
    So sqrt(3)/2 * x + 1/2 * y <= sqrt(3)/2 * S  -->  sqrt(3)x + y <= sqrt(3)S.
    
    Let's re-evaluate the standard inequalities for a regular hexagon with "radius" R (distance center to vertex) = S.
    The distance from center to side (apothem) is R * cos(30) = S * sqrt(3)/2.
    The sides are at distances S*sqrt(3)/2 from the center, in directions 0, 60, 120, 180, 240, 300.
    
    So the constraints are:
    For k in 0..5:
    dot(p, n_k) <= S * sqrt(3)/2
    where n_k = (cos(k*60), sin(k*60)).
    
    So S >= max_{p in P} max_{k} dot(p, n_k) / (sqrt(3)/2) = max_{p in P} max_{k} dot(p, n_k) * 2/sqrt(3).
    
    This is the correct formula.
    """
    # Transform inner polygons to local frame of outer hexagon
    rot_rad = math.radians(-rotation_deg)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    
    max_proj = 0.0
    
    for poly in inner_polys:
        # Translate to center
        translated = poly - np.array(center)
        # Rotate
        rotated_x = translated[:, 0] * cos_r - translated[:, 1] * sin_r
        rotated_y = translated[:, 0] * sin_r + translated[:, 1] * cos_r
        rotated_poly = np.column_stack((rotated_x, rotated_y))
        
        # Check projections against 6 normals
        for k in range(6):
            angle = k * math.pi / 3
            nx = math.cos(angle)
            ny = math.sin(angle)
            projections = rotated_poly[:, 0] * nx + rotated_poly[:, 1] * ny
            max_p = np.max(projections)
            if max_p > max_proj:
                max_proj = max_p
                
    # S * sqrt(3)/2 >= max_proj
    S = max_proj * 2.0 / math.sqrt(3.0)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial random configuration
    np.random.seed(42)
    
    # Inner hexagons: side length 1
    inner_side = 1.0
    
    # Initial positions: random in a square
    scale = 10.0
    inner = []
    for i in range(N):
        x = np.random.uniform(-scale, scale)
        y = np.random.uniform(-scale, scale)
        angle = np.random.uniform(0, 60)
        inner.append([x, y, angle])
        
    # Precompute vertices for initial configuration
    inner_polys = []
    for x, y, ang in inner:
        poly = _hexagon_vertices(x, y, inner_side, ang)
        inner_polys.append(poly)
        
    # Initial outer hexagon
    center = [0.0, 0.0]
    outer_rotation = 0.0
    
    # Calculate initial score
    # First, ensure no collisions? If collisions exist, we need to resolve or penalize.
    # Simulated Annealing with hard constraints is tricky. Let's use a penalty function.
    # Or, let's just use SA with a penalty for overlaps.
    
    def calculate_penalty(inner_polys, tol=1e-6):
        """
        Calculate penalty for overlapping inner hexagons.
        Returns 0 if no overlaps, otherwise sum of overlap areas (approximated by distance).
        """
        penalty = 0.0
        n = len(inner_polys)
        for i in range(n):
            for j in range(i+1, n):
                if not _polygons_disjoint(inner_polys[i], inner_polys[j]):
                    # Estimate overlap by distance between centers
                    c1 = np.mean(inner_polys[i], axis=0)
                    c2 = np.mean(inner_polys[j], axis=0)
                    dist = np.linalg.norm(c1 - c2)
                    # Minimum distance for disjoint is roughly 2 * apothem = sqrt(3)
                    # If dist < sqrt(3), they overlap.
                    # Penalty proportional to (sqrt(3) - dist)
                    min_dist = math.sqrt(3.0)
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
        return penalty
    
    def get_score(inner, center, outer_rotation):
        """
        Calculate the side length of the enclosing hexagon.
        If there are collisions, add a large penalty.
        """
        inner_polys = []
        for x, y, ang in inner:
            poly = _hexagon_vertices(x, y, inner_side, ang)
            inner_polys.append(poly)
            
        penalty = calculate_penalty(inner_polys)
        if penalty > 1e-6:
            return float('inf')
            
        S = _min_enclosing_hexagon_side(inner_polys, center, outer_rotation)
        return S
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    best_inner = [row[:] for row in inner]
    best_center = center[:]
    best_rotation = outer_rotation
    best_score = get_score(best_inner, best_center, best_rotation)
    
    current_inner = [row[:] for row in inner]
    current_center = center[:]
    current_rotation = outer_rotation
    current_score = best_score
    
    # Precompute vertices for current state to avoid recalculation in loop if possible?
    # It's hard to precompute because we move them.
    
    steps = 0
    while T > T_min and time.time() < deadline:
        steps += 1
        
        # Generate neighbor
        new_inner = [row[:] for row in current_inner]
        new_center = current_center[:]
        new_rotation = current_rotation
        
        # Randomly choose which parameter to perturb
        choice = np.random.random()
        
        if choice < 0.8:
            # Perturb one inner hexagon
            i = np.random.randint(N)
            dx = np.random.normal(0, 0.5)
            dy = np.random.normal(0, 0.5)
            dangle = np.random.normal(0, 10)
            new_inner[i][0] += dx
            new_inner[i][1] += dy
            new_inner[i][2] += dangle
            new_inner[i][2] %= 60  # Hexagon symmetry
        elif choice < 0.9:
            # Perturb center
            dx = np.random.normal(0, 0.5)
            dy = np.random.normal(0, 0.5)
            new_center[0] += dx
            new_center[1] += dy
        else:
            # Perturb rotation
            dangle = np.random.normal(0, 10)
            new_rotation += dangle
            new_rotation %= 60
            
        new_score = get_score(new_inner, new_center, new_rotation)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score < current_score:
            current_inner = new_inner
            current_center = new_center
            current_rotation = new_rotation
            current_score = new_score
            
            if new_score < best_score:
                best_inner = [row[:] for row in new_inner]
                best_center = new_center[:]
                best_rotation = new_rotation
                best_score = new_score
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if np.random.random() < prob:
                    current_inner = new_inner
                    current_center = new_center
                    current_rotation = new_rotation
                    current_score = new_score
                    
        T *= alpha
        
    return best_inner, best_center, best_score, best_rotation

# EVOLVE-BLOCK-END
