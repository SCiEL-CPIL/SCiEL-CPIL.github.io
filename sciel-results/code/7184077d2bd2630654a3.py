import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def min_distance_between_hexes(v1, v2):
        """
        Calculate the minimum distance between two convex polygons (hexagons).
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separate, returns the positive distance.
        We use this for constraints: dist > 0.
        
        For convex polygons, the minimum distance is either:
        1. Distance between two vertices.
        2. Distance between a vertex and an edge.
        
        We compute the minimum distance between any point in poly1 and any point in poly2.
        Since N=12, we have 66 pairs. Each pair has 6x6=36 vertex-vertex and 6x6*2=72 vertex-edge checks.
        This is fast enough for SLSQP if we vectorize or keep it tight.
        """
        min_dist = float('inf')
        
        # Helper: distance from point to segment
        def point_to_segment_dist(p, a, b):
            # Project p onto line ab
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = max(0, min(1, t))
            proj = a + t * ab
            return np.linalg.norm(p - proj)

        # Check vertex-vertex
        for p1 in v1:
            for p2 in v2:
                d = np.linalg.norm(p1 - p2)
                if d < min_dist:
                    min_dist = d
        
        # Check vertex-edge
        # Vertices of v1 to edges of v2
        for p in v1:
            for i in range(6):
                a = v2[i]
                b = v2[(i+1)%6]
                d = point_to_segment_dist(p, a, b)
                if d < min_dist:
                    min_dist = d
        
        # Vertices of v2 to edges of v1
        for p in v2:
            for i in range(6):
                a = v1[i]
                b = v1[(i+1)%6]
                d = point_to_segment_dist(p, a, b)
                if d < min_dist:
                    min_dist = d

        # If min_dist is very small, they might be overlapping.
        # To determine if they are actually overlapping (interior intersection), 
        # we can check if the minimum distance is 0 and if any vertex is inside the other.
        # However, for the constraint `g >= 0`, we want `min_dist > 0`.
        # If they overlap, min_dist will be 0 (or very close). 
        # We need a function that returns negative if overlapping.
        
        # Check for overlap explicitly if min_dist < 1e-9
        if min_dist < 1e-9:
             # Check if any vertex of v1 is inside v2
             def point_in_poly(pt, poly):
                 x, y = pt
                 inside = False
                 n = len(poly)
                 for i in range(n):
                     x1, y1 = poly[i]
                     x2, y2 = poly[(i + 1) % n]
                     if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                         inside = not inside
                 return inside
             
             for pt in v1:
                 if point_in_poly(pt, v2):
                     return -1.0 # Overlapping
             for pt in v2:
                 if point_in_poly(pt, v1):
                     return -1.0 # Overlapping
             
             # If not overlapping but touching, distance is 0.
             # For strict disjointness, we might need > 0, but touching is usually allowed in "disjoint" (interiors disjoint).
             # The problem says "disjoint", which usually means interiors do not overlap.
             # So dist >= 0 is valid.
             return 0.0
             
        return min_dist

    def constraints_func(params):
        """
        Returns array of constraint values.
        Each constraint is min_distance(pair_i, pair_j) >= 0.
        """
        positions = np.array(params).reshape(N, 2)
        cons = []
        for i in range(N):
            v1 = get_hex_verts(positions[i][0], positions[i][1], 1.0, 0.0)
            for j in range(i+1, N):
                v2 = get_hex_verts(positions[j][0], positions[j][1], 1.0, 0.0)
                d = min_distance_between_hexes(v1, v2)
                # We want d >= 0.
                # SLSQP constraints are of the form g(x) >= 0.
                # If d is negative, constraint is violated.
                cons.append(d)
        return cons

    def objective(params):
        """
        Objective: Minimize the outer side length.
        """
        if time.time() > deadline - 5:
            return 1e6
        positions = np.array(params).reshape(N, 2)
        angles = np.zeros(N)
        side = calc_min_outer_side(positions, angles, 0.0)
        return side

    # Strategy:
    # 1. Initialize with a hexagonal lattice configuration.
    # 2. Use SLSQP with constraints to enforce non-overlap.
    # 3. Optimize positions to minimize outer side.
    
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = sqrt3 * (i + j/2.0)
            y = 3 * j
            lattice_points.append((x, y))
            
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    
    x0 = init_positions.flatten()
    
    # Define constraints for SLSQP
    # SLSQP expects a list of dicts: {'type': 'ineq', 'fun': lambda x: g(x)}
    # Or we can pass a single function that returns an array if we use 'constraints' as a list.
    # Actually, SLSQP accepts a list of constraint dicts.
    # It is more efficient to define the constraints inside the loop or pass a function that returns all.
    # Standard SLSQP interface:
    # constraints = [{'type': 'ineq', 'fun': lambda x: g1(x)}, ...]
    
    # Since we have 66 constraints, creating 66 lambdas is fine, but passing a single function 
    # that returns a vector is not directly supported by standard SLSQP 'constraints' list 
    # unless we wrap it. However, SLSQP does support a list of dicts.
    # Let's try to optimize by only checking nearby pairs? No, 66 is small.
    
    # Alternative: Use 'SLSQP' with a single constraint function that returns a vector?
    # No, scipy SLSQP requires individual constraint functions in the list.
    # This might be slow due to overhead.
    
    # Let's try a different approach: Penalty method with a very large penalty, but using 
    # gradient information? No, SLSQP with constraints is the structural change requested.
    
    # To speed up SLSQP, we can reduce the number of constraints by only checking pairs 
    # that are "close" in the initial configuration? No, they can move.
    
    # Let's just run SLSQP.
    
    cons_list = []
    # Pre-compute pairs
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # To avoid creating 66 closures that capture loop variables incorrectly, 
    # we use default arguments.
    for i, j in pairs:
        def make_constraint(i, j):
            def constraint(x):
                if time.time() > deadline - 5:
                    return 1.0 # Dummy value to stop
                pos = np.array(x).reshape(N, 2)
                v1 = get_hex_verts(pos[i][0], pos[i][1], 1.0, 0.0)
                v2 = get_hex_verts(pos[j][0], pos[j][1], 1.0, 0.0)
                return min_distance_between_hexes(v1, v2)
            return constraint
        
        cons_list.append({'type': 'ineq', 'fun': make_constraint(i, j)})
        
    best_result = x0
    best_score = objective(x0)
    
    try:
        # SLSQP is good for small N with smooth constraints.
        # maxiter needs to be sufficient.
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            constraints=cons_list,
            options={'maxiter': 200, 'ftol': 1e-6, 'disp': False}
        )
        
        if res.success or res.fun < best_score:
             # Check if constraints are satisfied
             # res.fun is the objective value.
             # We should verify validity.
             final_pos = res.x.reshape(N, 2)
             valid = True
             polys = [get_hex_verts(final_pos[i][0], final_pos[i][1], 1.0, 0.0) for i in range(N)]
             for i in range(N):
                 for j in range(i+1, N):
                     if min_distance_between_hexes(polys[i], polys[j]) < -1e-6:
                         valid = False
                         break
                 if not valid: break
             
             if valid:
                 best_score = res.fun
                 best_result = res.x
    except Exception as e:
        pass

    # Fallback if SLSQP failed or returned invalid
    if best_score > 10.0: # Heuristic: if score is bad, use initial lattice
         best_result = x0
         best_score = objective(x0)

    final_positions = best_result.reshape(N, 2)
    final_angles = np.zeros(N)
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], 0.0])
    
    return inner, (0.0, 0.0), final_side, 0.0

