"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # 150s limit, keep 5s margin
    start_time = time.time()
    
    # 1. Geometry Helpers
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon."""
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def convex_hull(points):
        """Andrew's monotone chain convex hull algorithm."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        points = sorted(set(points))
        if len(points) <= 1:
            return points
        
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
            
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
            
        return lower[:-1] + upper[:-1]

    def min_enclosing_hexagon(points):
        """
        Find the minimum side length regular hexagon that encloses all points.
        We optimize over the angle of the hexagon.
        For a fixed angle, the minimum side length is determined by the maximum
        projection of the points onto the 3 normal directions of the hexagon.
        """
        # A regular hexagon with side length s and center (cx, cy) and angle theta
        # can be defined by 6 half-planes.
        # The distance from center to each side (apothem) is a = s * sqrt(3) / 2.
        # For the hexagon to contain a point p, the projection of (p - c) onto 
        # each of the 6 outward normals must be <= a.
        # This is equivalent to: max_i |dot(p - c, n_i)| <= a, where n_i are 
        # unit normals to the 6 sides.
        
        # We need to find c and theta such that s is minimized.
        # s = 2 * a / sqrt(3).
        # Minimizing s is equivalent to minimizing a.
        
        # Strategy:
        # 1. Translate points so their centroid is at (0,0).
        # 2. For a given angle theta, the optimal center c is the center of the
        #    minimum bounding box in the transformed coordinate system? No, it's the
        #    center of the minimum enclosing hexagon.
        #    Actually, for a fixed orientation, the problem reduces to finding the
        #    smallest hexagon (centered at some c) that contains the points.
        #    The side length s is determined by the maximum of the projections.
        #    Let the 6 normals be n_0...n_5.
        #    We need c such that for all p in points and all i:
        #    dot(p - c, n_i) <= a and dot(p - c, -n_i) <= a (which is dot(c-p, n_i) <= a)
        #    So: dot(c, n_i) >= dot(p, n_i) - a for all p
        #    And: dot(c, n_i) <= dot(p, n_i) + a for all p
        #    This implies:
        #    max_p dot(p, n_i) - a <= dot(c, n_i) <= min_p dot(p, n_i) + a
        #    For a solution c to exist, we must have:
        #    max_p dot(p, n_i) - min_p dot(p, n_i) <= 2a
        #    So a >= 0.5 * (max_p dot(p, n_i) - min_p dot(p, n_i))
        #    To minimize a, we choose a_i = 0.5 * (max_p dot(p, n_i) - min_p dot(p, n_i))
        #    The required a is max_i a_i.
        #    The center c can be chosen such that dot(c, n_i) is in the interval.
        #    Since the normals are linearly dependent (n_0 + n_1 + ... = 0), 
        #    we just need to check if the intersection of these strips is non-empty.
        #    For a regular hexagon, if the width constraints are satisfied, 
        #    the intersection is always non-empty?
        #    Yes, for centrally symmetric convex bodies like hexagons, 
        #    if the width in opposite directions is sufficient, there is a center.
        #    Specifically, c can be the centroid of the points? No.
        #    But we just need the minimal *side length*, which depends only on a.
        #    So s(theta) = (2/sqrt(3)) * max_i [ 0.5 * (max_p dot(p, n_i(theta)) - min_p dot(p, n_i(theta))) ]
        
        # We can sample theta.
        # The function s(theta) is periodic with period 60 degrees (pi/3).
        # It is also symmetric.
        
        best_s = float('inf')
        # Sample angles. The optimal angle is likely aligned with the convex hull edges.
        # We can do a coarse search followed by a fine search.
        
        # Centroid
        pts = np.array(points)
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid
        
        # Coarse search
        for theta_deg in range(0, 60, 1):
            theta = math.radians(theta_deg)
            # Normals for hexagon with angle theta
            # Vertices are at theta + i*pi/3.
            # Normals are at theta + i*pi/3 + pi/6?
            # Let's define normals to the sides.
            # Side i is perpendicular to normal n_i.
            # For a hexagon rotated by theta, the normals are at angles:
            # theta + pi/6 + i*pi/3 for i=0..5?
            # Let's check.
            # Vertex 0: (cos theta, sin theta).
            # Side 0 connects V0 and V1.
            # Normal to side 0 points outward.
            # Angle of V0 is theta. Angle of V1 is theta+60.
            # Midpoint angle is theta+30.
            # So normal 0 is at theta + 30 deg.
            # Yes.
            
            normals = []
            for i in range(6):
                ang = theta + math.pi/6 + i * math.pi/3
                normals.append((math.cos(ang), math.sin(ang)))
            
            max_width = 0
            for n in normals:
                # Project points onto n
                proj = pts_centered @ np.array(n)
                width = proj.max() - proj.min()
                if width > max_width:
                    max_width = width
            
            a = 0.5 * max_width
            s = 2 * a / math.sqrt(3)
            if s < best_s:
                best_s = s
                best_theta = theta

        # Fine search around best_theta
        for _ in range(20):
            for delta_deg in np.linspace(-1, 1, 21):
                theta = best_theta + math.radians(delta_deg)
                normals = []
                for i in range(6):
                    ang = theta + math.pi/6 + i * math.pi/3
                    normals.append((math.cos(ang), math.sin(ang)))
                
                max_width = 0
                for n in normals:
                    proj = pts_centered @ np.array(n)
                    width = proj.max() - proj.min()
                    if width > max_width:
                        max_width = width
                
                a = 0.5 * max_width
                s = 2 * a / math.sqrt(3)
                if s < best_s:
                    best_s = s
                    best_theta = theta

        # The center of the minimal hexagon for the optimal angle
        # We need to find c such that the hexagon is centered at c.
        # The condition was:
        # max_p dot(p, n_i) - a <= dot(c, n_i) <= min_p dot(p, n_i) + a
        # Let's pick the center as the average of the midpoints of the ranges?
        # Or just the centroid of the points?
        # Actually, the "width" calculation assumed the hexagon could slide.
        # The minimal enclosing hexagon centered at c has side length determined by
        # the maximum distance from c to the sides.
        # If we choose c = centroid, is it optimal?
        # Not necessarily. But for the purpose of *containing* the hexagons,
        # we can just return the side length and let the center be the centroid
        # of the convex hull of all vertices?
        # Wait, the task asks for the outer hexagon center.
        # If we use the centroid of the points, does the hexagon of side s contain them?
        # The width calculation ensures that there *exists* a center.
        # We can find one center by solving the linear inequalities.
        # For simplicity, we can use the centroid of the points as the center
        # and then *increase* s slightly if needed?
        # No, the width formula gives the *minimum* s for *some* center.
        # If we fix the center to be the centroid, s might need to be larger.
        # However, for a regular hexagon, the centroid of the points is often close
        # to the optimal center.
        # Let's refine: for the best theta, find the optimal center c.
        # This is a linear programming problem.
        # But we can approximate c as the centroid of the convex hull.
        
        # Let's just use the centroid of the input points as the center.
        # To be safe, we can calculate the required s for this specific center.
        # s_required = max_i ( max_p dot(p - c, n_i) ) / (sqrt(3)/2)
        # Let's do this for the best theta.
        
        theta = best_theta
        normals = []
        for i in range(6):
            ang = theta + math.pi/6 + i * math.pi/3
            normals.append((math.cos(ang), math.sin(ang)))
        
        c = centroid
        max_proj = 0
        for n in normals:
            # Project (p - c) onto n
            # Since we want the hexagon to contain p, we need dot(p-c, n) <= a
            # and dot(p-c, -n) <= a => dot(c-p, n) <= a
            # So a >= max_p |dot(p-c, n)|
            proj = (pts - c) @ np.array(n)
            max_abs = np.abs(proj).max()
            if max_abs > max_proj:
                max_proj = max_abs
        
        a = max_proj
        s = 2 * a / math.sqrt(3)
        
        # It's possible this s is slightly larger than the theoretical minimum
        # because c is fixed to centroid. But it's a valid containment.
        return s, c, theta

    def hexagons_disjoint(hex1, hex2, eps=1e-9):
        """Check if two hexagons are disjoint (non-overlapping interiors)."""
        # Get vertices
        v1 = np.array(get_hex_vertices(hex1[0], hex1[1], 1.0, hex1[2]))
        v2 = np.array(get_hex_vertices(hex2[0], hex2[1], 1.0, hex2[2]))
        
        # Check if any vertex of v1 is inside v2 or vice versa
        # And check for edge intersections
        # Simple robust check: use scipy.spatial? No, standard lib only.
        # We can use a point-in-polygon test for vertices.
        # If no vertex is inside, and no edges cross, they are disjoint.
        
        def point_in_poly(pt, poly):
            # Ray casting
            x, y = pt
            n = len(poly)
            inside = False
            p1x, p1y = poly[0]
            for i in range(1, n + 1):
                p2x, p2y = poly[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= min(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                        elif x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        def segments_intersect(p1, p2, p3, p4):
            # Check if segment p1-p2 intersects p3-p4
            def ccw(a, b, c):
                return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
            
            return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

        # Check vertices
        for v in v1:
            if point_in_poly(v, v2):
                return False
        for v in v2:
            if point_in_poly(v, v1):
                return False
        
        # Check edges
        for i in range(6):
            for j in range(6):
                if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return False
        
        return True

    # 2. Initial Configuration: Hexagonal Lattice
    # Arrange 12 hexagons in a hexagonal cluster.
    # Center + 6 surrounding + 5 more?
    # A hexagonal number 12 is not perfect. 1, 7, 19...
    # Let's use a 4x3 grid distorted into hex shape?
    # Or just a compact cluster.
    # Let's use a "flower" pattern: 1 in center, 6 around, 5 more around those.
    # Or simpler: a 3x4 grid with hexagonal offsets.
    
    # Let's generate a hexagonal lattice of 12 points.
    # Row 0: 4
    # Row 1: 4
    # Row 2: 4
    # With hex offsets.
    
    dx = 1.5
    dy = math.sqrt(3)
    
    positions = []
    # 4x3 grid
    for r in range(3):
        for c in range(4):
            x = c * dx + (r % 2) * dx/2
            y = r * dy
            positions.append([x, y, 0.0])
    
    # Center them
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    for p in positions:
        p[0] -= cx
        p[1] -= cy

    current = [p[:] for p in positions]
    
    # Calculate initial outer side
    all_vertices = []
    for p in current:
        vs = get_hex_vertices(p[0], p[1], 1.0, p[2])
        all_vertices.extend(vs)
    
    outer_side, outer_center, outer_angle = min_enclosing_hexagon(all_vertices)
    
    best_side = outer_side
    best_config = [p[:] for p in current]
    best_outer_center = outer_center
    best_outer_angle = outer_angle
    
    # 3. Simulated Annealing
    T = 1.0
    T_min = 1e-4
    alpha = 0.999
    step_size = 1.0
    
    while T > T_min and time.time() < deadline:
        # Perturb
        new_config = [p[:] for p in current]
        i = random.randint(0, N-1)
        
        # Move
        new_config[i][0] += random.gauss(0, step_size)
        new_config[i][1] += random.gauss(0, step_size)
        
        # Rotate
        if random.random() < 0.1:
            new_config[i][2] += random.gauss(0, 10)
        
        # Check disjointness
        valid = True
        for j in range(N):
            if j == i:
                continue
            if not hexagons_disjoint(new_config[i], new_config[j]):
                valid = False
                break
        
        if not valid:
            # Penalty
            delta_E = 100.0
        else:
            # Calculate new outer side
            all_vertices = []
            for p in new_config:
                vs = get_hex_vertices(p[0], p[1], 1.0, p[2])
                all_vertices.extend(vs)
            
            new_side, new_center, new_angle = min_enclosing_hexagon(all_vertices)
            delta_E = new_side - best_side
            
            if new_side < best_side:
                best_side = new_side
                best_config = [p[:] for p in new_config]
                best_outer_center = new_center
                best_outer_angle = new_angle
        
        # Accept/Reject
        if delta_E < 0:
            current = new_config
        else:
            if random.random() < math.exp(-delta_E / T):
                current = new_config
        
        T *= alpha
        
        # Adapt step size
        if random.random() < 0.01:
            step_size *= 0.95
            if step_size < 0.01:
                step_size = 0.01

    # Final check and return
    # Ensure best_config is valid and calculate final outer hexagon
    all_vertices = []
    for p in best_config:
        vs = get_hex_vertices(p[0], p[1], 1.0, p[2])
        all_vertices.extend(vs)
    
    outer_side, outer_center, outer_angle = min_enclosing_hexagon(all_vertices)
    
    return best_config, list(outer_center), outer_side, math.degrees(outer_angle)
# EVOLVE-BLOCK-END
