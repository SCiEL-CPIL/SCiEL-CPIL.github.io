import numpy as np
import time
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3.0)
    
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        angle = np.radians(angle_deg)
        return np.array([[cx + side*np.cos(angle + np.radians(60*k)),
                         cy + side*np.sin(angle + np.radians(60*k))] for k in range(6)])
    
    def point_in_hex(px, py, verts):
        n = len(verts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def hexagons_overlap(v1, v2):
        for vx, vy in v1:
            if point_in_hex(vx, vy, v2):
                return True
        for vx, vy in v2:
            if point_in_hex(vx, vy, v1):
                return True
        return False
    
    def min_outer_side(inner_hexes, center):
        cx, cy = center
        max_s = 0.0
        for hx, hy, ha in inner_hexes:
            verts = hex_vertices(hx, hy, ha)
            for vx, vy in verts:
                dx, dy = vx - cx, vy - cy
                dist = np.sqrt(dx*dx + dy*dy)
                ang = np.arctan2(dy, dx)
                phi_mod = ((ang + np.pi/6) % (np.pi/3)) - np.pi/6
                cos_val = max(np.cos(phi_mod), 1e-10)
                required_s = dist / cos_val
                if required_s > max_s:
                    max_s = required_s
        return max_s
    
    def make_arrangement(ring2_angles_deg):
        inner = [[0.0, 0.0, 0.0]]
        for i in range(6):
            a = np.radians(60.0 * i)
            inner.append([sqrt3*np.cos(a), sqrt3*np.sin(a), 0.0])
        for ang_deg in ring2_angles_deg:
            a = np.radians(ang_deg)
            inner.append([2*sqrt3*np.cos(a), 2*sqrt3*np.sin(a), 0.0])
        return inner
    
    def check_valid(inner):
        verts_list = [hex_vertices(x, y, a) for x, y, a in inner]
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                if hexagons_overlap(verts_list[i], verts_list[j]):
                    return False
        return True
    
    best_side = float('inf')
    best_inner = None
    
    all_angles = list(range(0, 360, 30))
    
    for combo in itertools.combinations(all_angles, 4):
        if time.time() > deadline - 10:
            break
        inner = make_arrangement(combo)
        if check_valid(inner):
            s = min_outer_side(inner, (0.0, 0.0))
            if s < best_side:
                best_side = s
                best_inner = inner
    
    if best_inner is None:
        best_inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        best_side = min_outer_side(best_inner, (0.0, 0.0))
    
    # Try rotating
    for rot in range(0, 30, 5):
        if time.time() > deadline - 5:
            break
        inner_rot = [[x, y, float(rot)] for x, y, _ in best_inner]
        if check_valid(inner_rot):
            s = min_outer_side(inner_rot, (0.0, 0.0))
            if s < best_side:
                best_side = s
                best_inner = inner_rot
    
    return best_inner, (0.0, 0.0), best_side, 0.0
