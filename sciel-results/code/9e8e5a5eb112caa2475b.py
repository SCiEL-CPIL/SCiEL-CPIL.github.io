import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 11
DEADLINE = time.time() + 135.0
H = np.sqrt(3.0) / 2.0

# Directions of a regular hexagon with a flat top (vertices at angles 0, 60, 120, 180, 240, 300)
DIRS = [(np.cos(np.radians(a)), np.sin(np.radians(a))) for a in range(0, 360, 60)]
# The 6 outward normals of those faces (perpendicular, pointing outward)
NORMALS = [(np.cos(np.radians(a)), np.sin(np.radians(a))) for a in range(0, 360, 60)]
# Check: for vertex at angle 0, the outward normal to the right face is at -30? 
# Actually for a regular hexagon centered at origin with vertex at angle 0 (1,0),
# the face to the "right" is vertical? No. Let me think again.
# Vertex at angle 0: (1,0). The two edges meeting there go to vertices at 60 and -60.
# The face between vertex 0 and vertex 60 has outward normal at 30 degrees.
# Let me use the 6 outward normals at angles 0, 60, 120, 180, 240, 300 for a "pointy-top" hex.
# Actually, let's define: a regular hexagon with flat top has vertices at 30, 90, 150, 210, 270, 330.
# Its outward normals (perpendicular to faces, pointing out) are at 0, 60, 120, 180, 240, 300.
# Let's use that convention.

# For the "flat top" hexagon (vertices at 30, 90, ...), the 6 supporting lines are at angles 0, 60, 120, 180, 240, 300.
# The side length S and the distance from center to each face (apothem) a = S * sqrt(3)/2.
# The distance from center to vertex = S.

# To compute the minimal enclosing regular hexagon of a given orientation theta:
# For a fixed orientation theta, the hexagon is defined by 6 half-planes.
# The hexagon with center c, side S, and orientation theta has faces with outward normals
# at angles theta + k*60 for k=0..5.
# The distance from center to each face is S * sqrt(3)/2.
# So for each normal direction n_i, we need: max_i ( (p_i - c) . n_i ) <= S * sqrt(3)/2
# where p_i are the vertices of the union of inner hexagons.
# To minimize S, for a fixed theta and c: S = (2/sqrt(3)) * max_i ( (p_i - c) . n_i )
# But we can also choose c to minimize the max. The optimal c for a fixed theta is the center of the
# minimum enclosing "hexagon" which is the intersection of 3 strips... 
# Actually, for a fixed orientation, the problem reduces to finding the smallest S such that there exists
# a center c with the 6 constraints. This is a linear programming problem, but we can simplify.
#
# For a fixed orientation theta, the hexagon is determined by 3 pairs of parallel supporting lines.
# The distance between each pair is 2*S*sqrt(3)/2 = S*sqrt(3).
# Let the 3 directions be at angles theta, theta+60, theta+120 (the normals).
# For direction i, let d_i = max_j (p_j . n_i) and l_i = min_j (p_j . n_i).
# The width in direction i is w_i = d_i - l_i. We need w_i <= S * sqrt(3) for all i.
# So S >= max_i (w_i / sqrt(3)).
# The center c should be at the midpoint: c . n_i = (d_i + l_i) / 2.
# Since the 3 normals span the space, c is determined.
# So for a fixed theta: S(theta) = (1/sqrt(3)) * max_i (w_i(theta)).
# Then we minimize over theta.

def compute_hexagon_vertices(cx, cy, S, theta_deg):
    """Vertices of a regular hexagon with center (cx,cy), side S, and flat-top orientation rotated by theta_deg."""
    theta = np.radians(theta_deg)
    # Flat-top hexagon has vertices at 30, 90, 150, 210, 270, 330 degrees
    angles = np.radians([30, 90, 150, 210, 270, 330]) + theta
    return np.array([[cx + S * np.cos(a), cy + S * np.sin(a)] for a in angles])

def get_inner_hexagon_vertices(x, y, angle_deg):
    """Vertices of a unit regular hexagon with center (x,y) and given rotation."""
    theta = np.radians(angle_deg)
    # Flat-top unit hexagon
    angles = np.radians([30, 90, 150, 210, 270, 330]) + theta
    return np.array([[x + np.cos(a), y + np.sin(a)] for a in angles])

def compute_min_enclosing_hexagon(all_vertices):
    """Compute minimal enclosing regular hexagon side length and center for given set of points.
    Returns (S, cx, cy, theta_deg) minimizing S over theta."""
    if len(all_vertices) == 0:
        return (1.0, 0.0, 0.0, 0.0)
    if len(all_vertices) == 1:
        return (1.0, all_vertices[0][0], all_vertices[0][1], 0.0)
    
    def S_of_theta(theta_deg):
        theta = np.radians(theta_deg)
        # Normals at theta + k*60
        max_proj = []
        min_proj = []
        for k in range(3):
            a = theta + np.radians(60 * k)
            nx, ny = np.cos(a), np.sin(a)
            projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            max_proj.append(np.max(projs))
            min_proj.append(np.min(projs))
        widths = [max_proj[i] - min_proj[i] for i in range(3)]
        return max(widths) / H  # H = sqrt(3)/2, so 1/H = 2/sqrt(3)
    
    # Minimize over theta
    best_S = float('inf')
    best_theta = 0.0
    # Try many angles
    for t in np.linspace(0, 60, 601):
        s = S_of_theta(t)
        if s < best_S:
            best_S = s
            best_theta = t
    # Refine
    for _ in range(50):
        step = 60.0 / (601 * 10)
        improved = False
        for dt in [-step, step]:
            t_new = best_theta + dt
            s = S_of_theta(t_new % 60)
            if s < best_S:
                best_S = s
                best_theta = t_new
                improved = True
        if not improved:
            break
    best_theta = best_theta % 60
    
    # Compute center
    theta = np.radians(best_theta)
    c = np.array([0.0, 0.0])
    for k in range(3):
        a = theta + np.radians(60 * k)
        nx, ny = np.cos(a), np.sin(a)
        projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
        c += 0.5 * (np.max(projs) + np.min(projs)) * np.array([nx, ny])
    
    return (best_S, c[0], c[1], best_theta)

def hexagon_overlap(v1, v2):
    """Check if two hexagons (given as vertex arrays) overlap using convex hull."""
    combined = np.vstack([v1, v2])
    if len(combined) < 3:
        return False
    try:
        hull = ConvexHull(combined)
        # Check if all vertices of v1 are inside the convex hull of v2 and vice versa
        # Simpler: check if the area of the convex hull of v1 U v2 is less than sum of areas
        # For regular hexagons, area = 3*sqrt(3)/2 * S^2
        # If they are disjoint, the convex hull area should be >= sum of areas (approximately)
        # But this is not exact. Let's use a different approach.
        return True  # Simplified: assume overlap if convex hulls intersect
    except:
        return False

def check_no_overlap(positions, angles):
    """Check that no two inner hexagons overlap. positions: list of (x,y), angles: list of angle_deg."""
    n = len(positions)
    vertices_list = []
    for i in range(n):
        x, y = positions[i]
        v = get_inner_hexagon_vertices(x, y, angles[i])
        vertices_list.append(v)
    for i in range(n):
        for j in range(i + 1, n):
            # Check if hexagon i and j overlap
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            # A point is inside a convex polygon if...
            # Use scipy's contains_points or manual check
            # For now, check if any vertex of v1 is inside v2 or any vertex of v2 is inside v1
            from scipy.spatial import Delaunay
            # Simpler: use the fact that two convex polygons overlap if one vertex of one is inside the other
            # or if an edge of one intersects an edge of the other
            # Let's use a point-in-polygon test
            if point_in_convex(v1[0], v2) or point_in_convex(v2[0], v1):
                return False
            # Also check edge intersections
            if edges_intersect(v1, v2):
                return False
    return True

def point_in_convex(p, poly):
    """Check if point p is inside convex polygon poly (CCW ordered)."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross < -1e-9:
            return False
    return True

def edges_intersect(poly1, poly2):
    """Check if any edge of poly1 intersects any edge of poly2."""
    n1 = len(poly1)
    n2 = len(poly2)
    for i in range(n1):
        a = poly1[i]
        b = poly1[(i + 1) % n1]
        for j in range(n2):
            c = poly2[j]
            d = poly2[(j + 1) % n2]
            if seg_intersect(a, b, c, d):
                return True
    return False

def seg_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def evaluate_configuration(positions, angles):
    """Evaluate a configuration: return (S, cx, cy, theta) or (inf, ...) if invalid."""
    if not check_no_overlap(positions, angles):
        return (float('inf'), 0.0, 0.0, 0.0)
    
    all_vertices = []
    for i, (x, y) in enumerate(positions):
        v = get_inner_hexagon_vertices(x, y, angles[i])
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    S, cx, cy, theta = compute_min_enclosing_hexagon(all_vertices)
    return (S, cx, cy, theta)

def generate_polyhexes(n):
    """Generate all free polyhexes of size n."""
    # This is complex. For n=11, there are many polyhexes.
    # Let's use a simpler approach: generate connected hexagonal clusters.
    # A hexagon can be represented by its axial coordinates (q, r).
    # Two hexagons are adjacent if their centers are distance 2 apart (for unit hexagons with side 1,
    # the distance between centers of adjacent hexagons is 2*sqrt(3)/2*... wait.
    # For unit regular hexagons, the distance between centers of two hexagons sharing a full edge is
    # 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    
    # Let's generate random connected clusters instead.
    clusters = set()
    
    # Start with a single hexagon and grow
    def grow(cluster, count):
        if count == n:
            # Normalize: translate so min q, min r is at origin
            qs = [c[0] for c in cluster]
            rs = [c[1] for c in cluster]
            min_q = min(qs)
            min_r = min(rs)
            normalized = frozenset((q - min_q, r - min_r) for q, r in cluster)
            clusters.add(normalized)
            return
        
        if len(cluster) > n:
            return
        
        # Get all adjacent positions
        adj = set()
        for q, r in cluster:
            for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                adj.add((q + dq, r + dr))
        adj -= set(cluster)
        
        for pos in adj:
            grow(cluster + [pos], count + 1)
    
    grow([(0, 0)], 1)
    return list(clusters)

def cluster_to_positions(cluster):
    """Convert axial coordinates to Cartesian positions for unit hexagons."""
    positions = []
    for q, r in cluster:
        # Axial to Cartesian for pointy-top hexagons
        x = H * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
    return positions

def refine_positions(positions, angles, max_iter=100, lr=0.01):
    """Refine positions to minimize the enclosing hexagon side, maintaining no overlap."""
    n = len(positions)
    best_S = float('inf')
    best_pos = list(positions)
    best_ang = list(angles)
    
    def objective(params):
        pos = [(params[2 * i], params[2 * i + 1]) for i in range(n)]
        ang = list(angles)
        result = evaluate_configuration(pos, ang)
        return result[0]
    
    # Simple gradient-free optimization
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        improved = False
        for i in range(n):
            for dx in [-lr, 0, lr]:
                for dy in [-lr, 0, lr]:
                    new_pos = list(best_pos)
                    new_pos[i] = (best_pos[i][0] + dx, best_pos[i][1] + dy)
                    result = evaluate_configuration(new_pos, best_ang)
                    if result[0] < best_S - 1e-6:
                        best_S = result[0]
                        best_pos = new_pos
                        improved = True
        if not improved:
            lr *= 0.8
            if lr < 1e-4:
                break
    
    return (best_S, best_pos, best_ang)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Generate some polyhexes
    if time.time() < deadline - 60:
        try:
            clusters = generate_polyhexes(N)
            # Limit to a reasonable number
            if len(clusters) > 200:
                clusters = clusters[:200]
        except Exception:
            clusters = []
    else:
        clusters = []
    
    if not clusters:
        # Fallback: linear arrangement
        positions = [(4.0 * k, 0.0) for k in range(-N // 2, N - N // 2)]
        angles = [0.0] * N
        S, cx, cy, theta = evaluate_configuration(positions, angles)
        return [[positions[i][0], positions[i][1], angles[i]] for i in range(N)], \
               (cx, cy), S, theta
    
    best_result = None
    best_S = float('inf')
    
    for cluster in clusters:
        if time.time() > deadline - 10:
            break
        
        positions = cluster_to_positions(list(cluster))
        angles = [0.0] * N
        
        # Try different global rotations
        for global_angle in [0.0, 30.0, 60.0]:
            rotated_positions = []
            for x, y in positions:
                ca = np.cos(np.radians(global_angle))
                sa = np.sin(np.radians(global_angle))
                rotated_positions.append((x * ca - y * sa, x * sa + y * ca))
            
            result = evaluate_configuration(rotated_positions, angles)
            if result[0] < best_S:
                best_S = result[0]
                best_result = (rotated_positions, angles, result[1], result[2], result[3])
        
        # Refine the best one for this cluster
        if best_result and best_result[0] == positions:
            if time.time() < deadline - 20:
                S, pos, ang = refine_positions(best_result[0], best_result[1], max_iter=50)
                if S < best_S:
                    best_S = S
                    result = evaluate_configuration(pos, ang)
                    best_result = (pos, ang, result[1], result[2], result[3])
    
    if best_result is None:
        # Fallback
        positions = [(4.0 * k, 0.0) for k in range(-N // 2, N - N // 2)]
        angles = [0.0] * N
        S, cx, cy, theta = evaluate_configuration(positions, angles)
        return [[positions[i][0], positions[i][1], angles[i]] for i in range(N)], \
               (cx, cy), S, theta
    
    positions, angles, cx, cy, theta = best_result
    inner = [[float(positions[i][0]), float(positions[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(cx), float(cy)), float(best_S), float(theta)

# EVOLVE-BLOCK-END
