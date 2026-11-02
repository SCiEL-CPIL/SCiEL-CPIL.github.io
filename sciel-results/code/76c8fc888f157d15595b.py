import math
import numpy as np
import time
import random

N = 11
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def get_hex_vertices(cx, cy, side, angle):
    """
    Get vertices of a regular hexagon.
    """
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_polygon(point, polygon):
    """
    Ray casting algorithm to check if a point is inside a polygon.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects segment p3-p4.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def polygons_intersect(poly1, poly2):
    """
    Check if two convex polygons intersect.
    """
    # Check if any vertex of poly1 is in poly2
    for p in poly1:
        if point_in_polygon(p, poly2):
            return True
    # Check if any vertex of poly2 is in poly1
    for p in poly2:
        if point_in_polygon(p, poly1):
            return True
    # Check if any edge of poly1 intersects any edge of poly2
    for i in range(len(poly1)):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % len(poly1)]
        for j in range(len(poly2)):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % len(poly2)]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial configuration: Hexagonal grid
    # Side length 1
    # Positions
    pos = []
    # Generate a small cluster
    # For N=11, a 2x2x2 cluster minus corners? Or a flower?
    # Let's use a known good starting point: a hexagon of side 2 has 7 hexagons.
    # Add 4 more around it.
    # Coordinates in axial system for side 1 hexagons
    axial_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2)
    ]
    
    # Convert axial to cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    for q, r in axial_coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pos.append([x, y, 0.0]) # x, y, angle

    # Outer hexagon initial guess
    # Center at centroid
    cx = np.mean([p[0] for p in pos])
    cy = np.mean([p[1] for p in pos])
    
    # Initial S: calculate required S for this configuration
    # Max distance from center to any vertex of any inner hex
    max_d = 0
    for p in pos:
        verts = get_hex_vertices(p[0], p[1], 1.0, p[2])
        for v in verts:
            d = math.hypot(v[0] - cx, v[1] - cy)
            if d > max_d:
                max_d = d
    # Apothem of outer hex S is S * sqrt(3)/2
    # So S * sqrt(3)/2 >= max_d  => S >= max_d * 2 / sqrt(3)
    S = max_d * 2.0 / math.sqrt(3.0)
    S *= 1.1 # Add margin
    
    # Outer rotation
    phi = 0.0

    # State: [x1, y1, a1, ..., xN, yN, aN, cx, cy, S, phi]
    state = []
    for p in pos:
        state.extend(p)
    state.extend([cx, cy, S, phi])
    
    # Objective function
    def evaluate(state):
        # Unpack
        pos_curr = []
        for i in range(N):
            x = state[i*3]
            y = state[i*3+1]
            a = state[i*3+2]
            pos_curr.append([x, y, a])
        
        cx = state[N*3]
        cy = state[N*3+1]
        S = state[N*3+2]
        phi = state[N*3+3]
        
        if S < 0:
            return float('inf')
        
        # Calculate vertices for all inner hexagons
        verts_list = []
        for p in pos_curr:
            verts = get_hex_vertices(p[0], p[1], 1.0, p[2])
            verts_list.append(verts)
        
        # Overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if polygons_intersect(verts_list[i], verts_list[j]):
                    # Calculate a rough penetration depth or just a fixed penalty
                    # Better: distance between centers
                    d = math.hypot(pos_curr[i][0]-pos_curr[j][0], pos_curr[i][1]-pos_curr[j][1])
                    # Min distance for disjoint aligned hexes is 2*apothem = sqrt(3)
                    # With rotation, it varies.
                    # Penalty proportional to how much they overlap?
                    # Let's use a simple penalty: 1000.0
                    overlap_penalty += 1000.0
                    # To make it smoother, we could compute min distance between boundaries, 
                    # but that's expensive.
        
        # Containment penalty
        contain_penalty = 0.0
        outer_verts = get_hex_vertices(cx, cy, S, phi)
        # Apothem of outer hex
        apothem_outer = S * math.sqrt(3) / 2.0
        
        for i in range(N):
            # Check if all vertices of inner hex i are inside outer hex
            for v in verts_list[i]:
                if not point_in_polygon(v, outer_verts):
                    # Calculate distance to boundary?
                    # Approximation: distance from center to vertex minus apothem
                    d_center = math.hypot(v[0]-cx, v[1]-cy)
                    # This is not accurate for rotated hexagons, but good enough for penalty
                    # Better: use point_in_polygon result
                    contain_penalty += 1000.0
                    break # One violation per hex is enough? Or sum all?
                    # Summing all might be better for gradient-like behavior
                    # But point_in_polygon is boolean.
        
        # Also check if inner hex center is too far?
        
        cost = S + overlap_penalty + contain_penalty
        return cost

    # Simulated Annealing
    T = 10.0
    T_min = 1e-6
    cooling = 0.999
    current_state = state[:]
    current_cost = evaluate(current_state)
    best_state = current_state[:]
    best_cost = current_cost
    
    t_start = time.time()
    while T > T_min and (time.time() - t_start) < (TIME_LIMIT - SAFETY_MARGIN):
        # Generate new state
        new_state = current_state[:]
        
        # Random move
        move_type = random.random()
        if move_type < 0.6:
            # Move a random inner hex
            i = random.randint(0, N-1)
            new_state[i*3] += random.gauss(0, 0.5)
            new_state[i*3+1] += random.gauss(0, 0.5)
            new_state[i*3+2] += random.gauss(0, 0.1)
        elif move_type < 0.8:
            # Move outer center
            new_state[N*3] += random.gauss(0, 0.5)
            new_state[N*3+1] += random.gauss(0, 0.5)
        elif move_type < 0.95:
            # Change S
            new_state[N*3+2] += random.gauss(0, 0.5)
            if new_state[N*3+2] < 1.0:
                new_state[N*3+2] = 1.0
        else:
            # Change phi
            new_state[N*3+3] += random.gauss(0, 0.1)
        
        new_cost = evaluate(new_state)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = new_state
            current_cost = new_cost
            if current_cost < best_cost:
                best_state = current_state[:]
                best_cost = current_cost
        
        T *= cooling
        
        # Cooling rate adjustment if stuck?
        # Keep it simple for now.
        
    # Extract best solution
    pos_best = []
    for i in range(N):
        x = best_state[i*3]
        y = best_state[i*3+1]
        a = best_state[i*3+2]
        pos_best.append([x, y, a])
    
    cx = best_state[N*3]
    cy = best_state[N*3+1]
    S = best_state[N*3+2]
    phi = best_state[N*3+3]
    
    # Verify and adjust S to be minimal valid
    # The SA might have found a valid configuration with S slightly larger than needed
    # We can tighten S
    # Find max distance from (cx, cy) to any vertex of any inner hex
    max_d = 0
    for p in pos_best:
        verts = get_hex_vertices(p[0], p[1], 1.0, p[2])
        for v in verts:
            d = math.hypot(v[0] - cx, v[1] - cy)
            if d > max_d:
                max_d = d
    
    # The minimal S for a given orientation phi and center (cx,cy) is determined by the support function
    # For a regular hexagon, the minimal S such that a set of points is inside is:
    # S = 2 / sqrt(3) * max_k ( max_i ( (x_i-cx)cos(theta_k) + (y_i-cy)sin(theta_k) ) )
    # where theta_k are the normals to the sides.
    # Normals angles: phi + pi/6 + k*pi/3
    
    min_S = 0.0
    for k in range(6):
        angle = phi + math.pi/6 + k * math.pi/3
        nx = math.cos(angle)
        ny = math.sin(angle)
        max_proj = -float('inf')
        for p in pos_best:
            verts = get_hex_vertices(p[0], p[1], 1.0, p[2])
            for v in verts:
                proj = (v[0]-cx)*nx + (v[1]-cy)*ny
                if proj > max_proj:
                    max_proj = proj
        # Apothem = S * sqrt(3)/2
        # max_proj <= Apothem
        # S >= 2 * max_proj / sqrt(3)
        S_req = 2.0 * max_proj / math.sqrt(3.0)
        if S_req > min_S:
            min_S = S_req
            
    # Use the tighter S
    S_final = min_S
    
    # Final check for overlaps?
    # If SA found a valid config, it should be fine.
    # But if we reduced S, we might have introduced overlaps? No, S reduction doesn't affect inner positions.
    # It only changes the container.
    
    # Return
    inner = []
    for p in pos_best:
        inner.append([float(p[0]), float(p[1]), float(math.degrees(p[2]))])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(math.degrees(phi))
    
    return inner, center, side, rotation
