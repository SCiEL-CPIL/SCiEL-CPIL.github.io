import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12
SQRT3 = math.sqrt(3.0)
APOTHEM = SQRT3 / 2.0

def hex_vertices(center, side, angle):
    """
    Calculate the vertices of a regular hexagon.
    center: (x, y)
    side: side length
    angle: rotation in radians
    """
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a convex polygon.
    polygon: Nx2 array of vertices.
    """
    x, y = point
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross < -1e-9:
            return False
    return True

def polygons_intersect(poly1, poly2):
    """
    Check if two convex polygons intersect.
    Uses Separating Axis Theorem (SAT).
    """
    def project_polygon(axis, poly):
        projections = np.dot(poly, axis)
        return np.min(projections), np.max(projections)

    def get_normals(poly):
        normals = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm
            normals.append(normal)
        return normals

    axes = get_normals(poly1) + get_normals(poly2)
    
    for axis in axes:
        min1, max1 = project_polygon(axis, poly1)
        min2, max2 = project_polygon(axis, poly2)
        
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
            
    return True

def check_validity(inner, outer_center, outer_side, outer_angle):
    """
    Check if inner hexagons are disjoint and inside the outer hexagon.
    """
    if outer_side <= 0:
        return False, np.inf

    outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
    n_inner = len(inner)
    inner_verts = []
    
    for i in range(n_inner):
        x, y, ang = inner[i]
        v = hex_vertices((x, y), 1.0, np.radians(ang))
        inner_verts.append(v)
        
        # Check containment
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                return False, np.inf

    # Check non-overlap
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            if polygons_intersect(inner_verts[i], inner_verts[j]):
                return False, np.inf
                
    return True, 0.0

def distance_point_to_segment(p, a, b):
    """
    Distance from point p to segment ab.
    """
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-12:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_len_sq
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def min_distance_to_polygon_boundary(point, polygon):
    """
    Calculate minimum distance from a point to the boundary of a convex polygon.
    """
    min_d = np.inf
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        d = distance_point_to_segment(point, p1, p2)
        if d < min_d:
            min_d = d
    return min_d

def min_distance_between_polygons(poly1, poly2):
    """
    Approximate minimum distance between two convex polygons.
    Returns negative value if intersecting (heuristic depth).
    """
    if polygons_intersect(poly1, poly2):
        # Estimate depth: min distance from any vertex of poly1 to poly2 boundary
        min_d = np.inf
        for v in poly1:
            d = min_distance_to_polygon_boundary(v, poly2)
            if d < min_d:
                min_d = d
        for v in poly2:
            d = min_distance_to_polygon_boundary(v, poly1)
            if d < min_d:
                min_d = d
        # If intersecting, min_d is small. We want a strong penalty.
        # Using a fixed negative value plus the small distance ensures gradient direction.
        return -1.0 - min_d
        
    min_d = np.inf
    for v in poly1:
        d = min_distance_to_polygon_boundary(v, poly2)
        if d < min_d:
            min_d = d
    for v in poly2:
        d = min_distance_to_polygon_boundary(v, poly1)
        if d < min_d:
            min_d = d
    return min_d

def penalty_function(params):
    """
    Objective function for optimization.
    params: 
        - outer_side (1)
        - outer_center (2)
        - outer_angle (1)
        - inner centers (2*N)
        - inner angles (N)
    """
    outer_side = params[0]
    outer_cx, outer_cy = params[1:3]
    outer_angle = params[3]
    
    inner_params = params[4:]
    inner = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        a = inner_params[3*i+2]
        inner.append((x, y, a))
        
    outer_center = (outer_cx, outer_cy)
    
    # Compute vertices for all inner hexagons
    inner_verts = []
    for i in range(N):
        x, y, a = inner[i]
        v = hex_vertices((x, y), 1.0, np.radians(a))
        inner_verts.append(v)
        
    outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
    
    total_penalty = 0.0
    
    # 1. Containment penalty
    # For each inner vertex, if outside outer polygon, add distance squared
    for v in inner_verts:
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                dist = min_distance_to_polygon_boundary(pt, outer_verts)
                # Heavy penalty for containment violation
                total_penalty += dist * dist * 100.0

    # 2. Overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            min_dist = min_distance_between_polygons(inner_verts[i], inner_verts[j])
            if min_dist < 0:
                # Intersecting
                # min_dist is negative. Penalty should be positive and large.
                # If min_dist = -1.05, penalty = 1.05 * 100 = 105.
                total_penalty += (-min_dist) * 100.0
            elif min_dist < 1e-3:
                # Very close, add small penalty to encourage separation
                total_penalty += (1e-3 - min_dist) * 10.0

    # Objective: Minimize outer_side + penalties
    return outer_side + total_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    best_score = np.inf
    best_params = None
    
    # Initial configurations
    def init_hex_cluster():
        """
        Standard hexagonal packing initialization.
        """
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        # Center
        idx = 0
        params[4 + 3*idx] = 0
        params[4 + 3*idx+1] = 0
        params[4 + 3*idx+2] = 0
        
        # 6 neighbors at distance 2*APOTHEM * 2 = 2*SQRT3? 
        # Distance between centers of touching hexagons is 2 * apothem * 2? No.
        # Distance between centers of two touching unit hexagons is 2 * apothem * 2?
        # Apothem = SQRT3/2. Distance = 2 * Apothem = SQRT3? No.
        # If they share an edge, the distance between centers is 2 * Apothem = SQRT3.
        # Wait, if side=1, apothem=SQRT3/2. Distance between parallel sides is SQRT3.
        # So center-to-center distance for touching is SQRT3.
        
        # Let's verify.
        # Hex 1 at (0,0). Rightmost vertex (1,0).
        # Hex 2 touching on the right. Its leftmost vertex must be (1,0)?
        # Leftmost vertex of Hex 2 at (x,0) is (x-1, 0).
        # So x-1 = 1 => x=2.
        # So distance is 2?
        # Let's check the shape.
        # Vertices at angles 0, 60, 120...
        # (1,0), (0.5, SQRT3/2), (-0.5, SQRT3/2), (-1,0)...
        # The edge from (0.5, SQRT3/2) to (1,0) is not vertical.
        # The "flat" side is not vertical.
        # The distance between centers of two hexagons sharing a vertical edge?
        # There are no vertical edges in this orientation.
        # Edges are at 30, 90, 150... degrees?
        # Midpoint of edge 0-60: (0.75, SQRT3/4).
        # Normal to this edge: (SQRT3/2, 0.5) ?
        # Vector (0.5, SQRT3/2) - (1,0) = (-0.5, SQRT3/2).
        # Normal: (SQRT3/2, 0.5).
        # Distance from center (0,0) to this edge:
        # Line: SQRT3/2 x + 0.5 y = C.
        # Plug in (1,0): C = SQRT3/2.
        # Distance = |C| / sqrt(3/4 + 1/4) = SQRT3/2.
        # So apothem is SQRT3/2.
        # If two hexagons share this edge, their centers are at distance 2 * apothem = SQRT3.
        
        dist = SQRT3
        for k in range(6):
            idx = k + 1
            ang = np.radians(k * 60)
            params[4 + 3*idx] = dist * np.cos(ang)
            params[4 + 3*idx+1] = dist * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        # 5 more
        # Place them in the next ring
        for k in range(5):
            idx = k + 7
            ang = np.radians(k * 72 + 30)
            # Distance for second ring is roughly 2 * dist
            d = 2.0 * dist
            params[4 + 3*idx] = d * np.cos(ang)
            params[4 + 3*idx+1] = d * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        return params

    def init_grid():
        """
        Grid initialization with large spacing.
        """
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        for i in range(N):
            col = i % 4
            row = i // 4
            ix = col * 3.5
            iy = row * 3.0
            params[4 + 3*i] = ix
            params[4 + 3*i+1] = iy
            params[4 + 3*i+2] = 0.0
        return params

    initializations = [init_hex_cluster(), init_grid()]
    
    # Add some random perturbations to initializations to escape local minima
    np.random.seed(42)
    for _ in range(2):
        base = init_hex_cluster()
        perturbed = base.copy()
        # Perturb inner positions slightly
        for i in range(N):
            perturbed[4 + 3*i] += np.random.normal(0, 0.5)
            perturbed[4 + 3*i+1] += np.random.normal(0, 0.5)
        initializations.append(perturbed)
        
    for init_params in initializations:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead for robustness with non-smooth penalties
            res = minimize(penalty_function, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except Exception:
            pass
            
    if best_params is None:
        # Fallback: return a valid but large configuration
        inner = []
        for i in range(N):
            col = i % 4
            row = i // 4
            x = col * 3.5 - 3.5*1.5
            y = row * 3.0 - 3.0*1.0
            inner.append((x, y, 0))
            
        return inner, (0, 0), 10.0, 0.0

    # Extract best params
    outer_side = best_params[0]
    outer_cx = best_params[1]
    outer_cy = best_params[2]
    outer_angle = best_params[3]
    
    inner = []
    for i in range(N):
        x = best_params[4 + 3*i]
        y = best_params[4 + 3*i+1]
        a = best_params[4 + 3*i+2]
        inner.append((x, y, a))
        
    outer_center = (outer_cx, outer_cy)
    
    # Final verification
    valid, _ = check_validity(inner, outer_center, outer_side, outer_angle)
    
    if not valid:
        # If the optimized result is invalid, fall back to a simple valid configuration.
        inner = []
        for i in range(N):
            col = i % 4
            row = i // 4
            x = col * 3.5 - 5.0
            y = row * 3.0 - 1.5
            inner.append((x, y, 0))
        outer_side = 10.0
        outer_center = (0, 0)
        outer_angle = 0.0
        
    return inner, outer_center, outer_side, outer_angle
