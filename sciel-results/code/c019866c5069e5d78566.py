import numpy as np
import time
from scipy.spatial.distance import cdist

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Helper: Generate vertices of a regular hexagon
    def hex_verts(cx, cy, s, rot):
        # Regular hexagon vertices relative to center
        # Flat-top or pointy-top? 
        # Standard math: angle k * 60 deg.
        # If angle=0, vertex at (s, 0). This is a "pointy-right" hexagon.
        # The container is also a regular hexagon.
        # To ensure "unit regular hexagons", side length s=1.
        # Vertices at distance s from center.
        angles = np.deg2rad(np.arange(6) * 60 + rot)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_dist_polygons(v1, v2):
        """
        Calculate minimum distance between two convex polygons defined by vertices.
        If they intersect, return 0.
        """
        # Use shapely if available? No, stdlib + numpy only.
        # Simple approach: check if any vertex of v1 is inside v2 or vice versa, 
        # or if any edges intersect.
        # If no intersection, min dist is min distance between any pair of edges or vertices.
        
        def point_in_poly(pt, poly):
            # Ray casting
            x, y = pt
            n = len(poly)
            inside = False
            p1x, p1y = poly[0]
            for i in range(n + 1):
                p2x, p2y = poly[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        def segments_intersect(p1, p2, p3, p4):
            # Check if segment p1-p2 intersects p3-p4
            # Using cross products
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        # 1. Check for intersection
        for i in range(6):
            if point_in_poly(v1[i], v2):
                return 0.0
        for i in range(6):
            if point_in_poly(v2[i], v1):
                return 0.0
            
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return 0.0

        # 2. Calculate min distance
        min_d = float('inf')
        # Distance between vertices and edges
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                
                # Point p3 to line p1-p2
                # Vector p1->p2
                v = p2 - p1
                w = p3 - p1
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p3 - p1)
                elif c1 >= c2:
                    d = np.linalg.norm(p3 - p2)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d: min_d = d
                
                # Point p4 to line p1-p2
                w = p4 - p1
                c1 = np.dot(w, v)
                if c1 <= 0:
                    d = np.linalg.norm(p4 - p1)
                elif c1 >= c2:
                    d = np.linalg.norm(p4 - p2)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d: min_d = d

        return min_d

    def evaluate_solution(params):
        """
        params: [x1, y1, a1, ..., x12, y12, a12, outer_s, outer_rot]
        Returns: (outer_side_length, validity)
        """
        # Unpack
        xs = params[0:12]
        ys = params[12:24]
        as_ = params[24:36]
        outer_s = params[36]
        outer_rot = params[37]
        
        # Generate inner hexagons
        inner_verts = []
        for i in range(12):
            v = hex_verts(xs[i], ys[i], 1.0, as_[i])
            inner_verts.append(v)
            
        # Check disjointness
        for i in range(12):
            for j in range(i+1, 12):
                d = min_dist_polygons(inner_verts[i], inner_verts[j])
                if d < 1e-5: # Overlapping
                    return float('inf'), False

        # Check containment in outer hexagon
        # Outer hexagon center (0,0), side length outer_s, rotation outer_rot
        # Vertices of outer hexagon are at distance outer_s from center.
        # A point (x,y) is inside a regular hexagon centered at origin with side s
        # if its distance to the center is <= s * cos(angle_diff) for the closest vertex/edge?
        # Actually, the inradius of a hexagon with side s is r = s * sqrt(3)/2.
        # The apothem is r.
        # The condition for a point (x,y) to be inside a hexagon with rotation rot:
        # The hexagon is the intersection of 6 half-planes.
        # Normal vectors to the edges are at angles rot + 30 + k*60? 
        # Let's check vertex angles: 0, 60, 120...
        # Edge midpoints (apothems) are at 30, 90, 150...
        # So the supporting lines are at distance r = s*sqrt(3)/2 from origin,
        # with normals at angles 30 + k*60 + rot.
        
        r = outer_s * np.sqrt(3) / 2.0
        normals = np.deg2rad(30 + np.arange(6)*60 + outer_rot)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(12):
            for v in inner_verts[i]:
                # Project point onto all normal vectors
                # The distance from origin to the edge along the normal is r.
                # If dot(v, n) > r, it's outside.
                dists = np.dot(v, n_vecs.T)
                if np.any(dists > r + 1e-5):
                    return float('inf'), False

        return outer_s, True

    def objective(params):
        """
        Objective for optimization.
        We want to minimize outer_s, subject to constraints.
        We can use a penalty method or just check validity.
        For SA, we define a "cost" that is outer_s if valid, else large penalty.
        """
        val, valid = evaluate_solution(params)
        if not valid:
            return 1000.0 + val # Penalty
        return val

    # Initialize
    # Start with a large outer hexagon
    best_params = None
    best_val = float('inf')
    
    # Random initialization
    np.random.seed(42)
    # Initial positions spread out
    init_params = np.zeros(38)
    # Place them in a circle roughly
    for i in range(12):
        angle = i * 30
        r = 3.0
        init_params[i] = r * np.cos(np.deg2rad(angle))
        init_params[i+12] = r * np.sin(np.deg2rad(angle))
        init_params[i+24] = 0.0 # Random rotation
    init_params[36] = 10.0 # Large outer side
    init_params[37] = 0.0
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.99
    current_params = init_params.copy()
    current_val, current_valid = evaluate_solution(current_params)
    if current_valid:
        best_params = current_params.copy()
        best_val = current_val
        
    steps = 0
    max_steps = 50000
    
    while steps < max_steps:
        if time.time() > deadline:
            break
            
        # Perturbation
        new_params = current_params.copy()
        # Move one inner hexagon
        idx = np.random.randint(12)
        new_params[idx] += np.random.normal(0, 0.1)
        new_params[idx+12] += np.random.normal(0, 0.1)
        new_params[idx+24] += np.random.normal(0, 10) # degrees
        
        # Occasionally adjust outer size/rotation
        if np.random.rand() < 0.1:
            new_params[36] *= (1 + np.random.normal(0, 0.01))
            if new_params[36] < 2.0: new_params[36] = 2.0
            new_params[37] += np.random.normal(0, 5)
            
        new_val, new_valid = evaluate_solution(new_params)
        
        # Acceptance criterion
        if new_val < current_val:
            current_params = new_params
            current_val = new_val
            if new_valid and current_val < best_val:
                best_params = new_params.copy()
                best_val = current_val
        else:
            delta = new_val - current_val
            if np.random.rand() < np.exp(-delta / T):
                current_params = new_params
                current_val = new_val
                
        T *= cooling
        steps += 1
        
    # If we didn't find a valid solution, return a safe large one
    if best_params is None:
         # Fallback to grid if SA failed completely (unlikely with large start)
         inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
         return inner, (0.0, 0.0), 40.0, 0.0

    # Extract best solution
    xs = best_params[0:12]
    ys = best_params[12:24]
    as_ = best_params[24:36]
    outer_s = best_params[36]
    outer_rot = best_params[37]
    
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], as_[i]])
        
    return inner, (0.0, 0.0), outer_s, outer_rot
