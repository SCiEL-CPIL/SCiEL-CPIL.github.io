import math
import time
import random
import numpy as np

def _hex_vertices(cx, cy, s, angle_deg):
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _on_segment(p, q, r):
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
        min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False

def _segments_intersect(p1, p2, p3, p4):
    o1 = _cross(p1, p2, p3)
    o2 = _cross(p1, p2, p4)
    o3 = _cross(p3, p4, p1)
    o4 = _cross(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(p1, p3, p2): return True
    if o2 == 0 and _on_segment(p1, p4, p2): return True
    if o3 == 0 and _on_segment(p3, p1, p4): return True
    if o4 == 0 and _on_segment(p3, p2, p4): return True

    return False

def _min_dist_point_seg(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx*abx + aby*aby
    if ab_len_sq == 0:
        return math.hypot(px-ax, py-ay)
    t = ((px-ax)*abx + (py-ay)*aby) / ab_len_sq
    if t < 0:
        t = 0
    elif t > 1:
        t = 1
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px-cx, py-cy)

def _min_dist_poly(poly1, poly2):
    # Returns min distance between two convex polygons.
    # Negative if overlapping (approx overlap depth).
    min_d = float('inf')
    
    # Check if any vertex of poly1 is inside poly2
    def point_in_poly(pt, poly):
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i+1)%n]
            cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
            if cross < -1e-9:
                return False
        return True

    # Note: This assumes CCW. We will ensure CCW in _hex_vertices.
    # _hex_vertices generates CCW if angle increases.
    
    for v in poly1:
        if point_in_poly(v, poly2):
            return -1.0 # Overlapping
    
    for v in poly2:
        if point_in_poly(v, poly1):
            return -1.0 # Overlapping

    # Check edge intersections
    for i in range(len(poly1)):
        a1 = poly1[i]
        b1 = poly1[(i+1)%len(poly1)]
        for j in range(len(poly2)):
            a2 = poly2[j]
            b2 = poly2[(j+1)%len(poly2)]
            if _segments_intersect(a1, b1, a2, b2):
                return -1.0 # Overlapping

    # Min distance between edges
    for i in range(len(poly1)):
        a1 = poly1[i]
        b1 = poly1[(i+1)%len(poly1)]
        for j in range(len(poly2)):
            a2 = poly2[j]
            b2 = poly2[(j+1)%len(poly2)]
            # Dist between segments
            d1 = _min_dist_point_seg(a1, a2, b2)
            d2 = _min_dist_point_seg(b1, a2, b2)
            d3 = _min_dist_point_seg(a2, a1, b1)
            d4 = _min_dist_point_seg(b2, a1, b1)
            min_d = min(min_d, d1, d2, d3, d4)
            
    return min_d

def _dist_point_to_poly(pt, poly):
    # Min distance from point to polygon boundary.
    # If inside, return 0 (or negative? No, we want violation if outside).
    # We want to know how far outside it is.
    # If inside, dist is 0.
    # If outside, dist is min dist to any edge.
    
    # Check inside
    n = len(poly)
    inside = True
    for i in range(n):
        a = poly[i]
        b = poly[(i+1)%n]
        cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
        if cross < -1e-9:
            inside = False
            break
            
    if inside:
        return 0.0
        
    min_d = float('inf')
    for i in range(n):
        a = poly[i]
        b = poly[(i+1)%n]
        d = _min_dist_point_seg(pt, a, b)
        min_d = min(min_d, d)
        
    return min_d

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    N = 11
    
    # Initialize positions in hex lattice
    q_r_coords = [(0,0)]
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    q_r_coords.extend(ring1)
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)]
    ring2_sorted = sorted(ring2, key=lambda c: c[0]**2 + c[1]**2 + c[0]*c[1])
    q_r_coords.extend(ring2_sorted[:4])
    
    initial_positions = []
    for q, r in q_r_coords:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        initial_positions.append([x, y, 0.0])
        
    # Center
    xs = [p[0] for p in initial_positions]
    ys = [p[1] for p in initial_positions]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    for p in initial_positions:
        p[0] -= cx
        p[1] -= cy
        
    # Initial State
    state = {
        'inner': [p[:] for p in initial_positions],
        'outer_cx': 0.0,
        'outer_cy': 0.0,
        'outer_S': 10.0, # Start large
        'outer_ang': 0.0
    }
    
    # Find initial valid S
    def calc_cost(s):
        # Returns (S + Penalty)
        outer_verts = _hex_vertices(s['outer_cx'], s['outer_cy'], s['outer_S'], s['outer_ang'])
        penalty = 0.0
        
        inner_verts_list = []
        for x, y, ang in s['inner']:
            verts = _hex_vertices(x, y, 1.0, ang)
            inner_verts_list.append(verts)
            
        # Check boundaries
        for verts in inner_verts_list:
            for v in verts:
                d = _dist_point_to_poly(v, outer_verts)
                if d > 0:
                    penalty += d*d
                    
        # Check overlaps
        for i in range(len(inner_verts_list)):
            for j in range(i+1, len(inner_verts_list)):
                d = _min_dist_poly(inner_verts_list[i], inner_verts_list[j])
                if d < 0:
                    # Overlap depth approx -d? 
                    # _min_dist_poly returns -1.0 on overlap. 
                    # We need a better metric for SA.
                    # Let's just use a large penalty.
                    penalty += 10.0
                    
        return s['outer_S'] + penalty

    # Initial S search
    current_S = 5.0
    while current_S < 20:
        state['outer_S'] = current_S
        cost = calc_cost(state)
        if cost < current_S + 1e-6: # Valid
            break
        current_S += 0.1
    state['outer_S'] = current_S
    
    best_state = {
        'inner': [p[:] for p in state['inner']],
        'outer_cx': state['outer_cx'],
        'outer_cy': state['outer_cy'],
        'outer_S': state['outer_S'],
        'outer_ang': state['outer_ang']
    }
    best_cost = calc_cost(best_state)
    
    T = 1.0
    T_min = 0.001
    
    while time.time() < deadline:
        # Generate perturbation
        new_state = {
            'inner': [p[:] for p in state['inner']],
            'outer_cx': state['outer_cx'],
            'outer_cy': state['outer_cy'],
            'outer_S': state['outer_S'],
            'outer_ang': state['outer_ang']
        }
        
        # Random move
        move_type = random.randint(0, 4)
        if move_type == 0:
            # Move inner
            idx = random.randint(0, N-1)
            new_state['inner'][idx][0] += random.gauss(0, 0.1)
            new_state['inner'][idx][1] += random.gauss(0, 0.1)
        elif move_type == 1:
            # Rotate inner
            idx = random.randint(0, N-1)
            new_state['inner'][idx][2] += random.gauss(0, 5.0)
        elif move_type == 2:
            # Move outer
            new_state['outer_cx'] += random.gauss(0, 0.1)
            new_state['outer_cy'] += random.gauss(0, 0.1)
        elif move_type == 3:
            # Rotate outer
            new_state['outer_ang'] += random.gauss(0, 5.0)
        else:
            # Shrink S
            new_state['outer_S'] *= (1.0 - random.uniform(0.0, 0.05))
            
        new_cost = calc_cost(new_state)
        delta = new_cost - best_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            state = new_state
            if new_cost < best_cost:
                best_cost = new_cost
                best_state = {
                    'inner': [p[:] for p in state['inner']],
                    'outer_cx': state['outer_cx'],
                    'outer_cy': state['outer_cy'],
                    'outer_S': state['outer_S'],
                    'outer_ang': state['outer_ang']
                }
                
        T *= 0.995
        if T < T_min:
            T = T_min
            
    # Return best
    return [
        [best_state['inner'][i][0], best_state['inner'][i][1], best_state['inner'][i][2]] 
        for i in range(N)
    ], [best_state['outer_cx'], best_state['outer_cy']], best_state['outer_S'], best_state['outer_ang']

# EVOLVE-BLOCK-END
