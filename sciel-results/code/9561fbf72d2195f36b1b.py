import numpy as np
import time
import math

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    normals = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    h_in = math.sqrt(3) / 2
    
    def get_vertices(x, y, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        verts = (R @ hex_verts.T).T
        verts += np.array([x, y])
        return verts

    def point_in_hex(p, center, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        p_local = R @ (p - center)
        projs = p_local @ normals.T
        return np.all(projs <= h_in + 1e-9)

    def segments_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects p3-p4
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def check_overlap(v1, v2):
        # v1, v2 are (6,2) arrays
        # Check vertices
        for p in v1:
            if point_in_hex(p, v2[0] - (v2[0] - v2[0]), 0): # Wrong center/angle
                pass
        # Need center and angle for point_in_hex.
        # Let's pass center and angle explicitly.
        return False # Placeholder

    def check_overlap_full(pos1, ang1, pos2, ang2):
        c1 = np.array(pos1)
        c2 = np.array(pos2)
        v1 = get_vertices(pos1[0], pos1[1], ang1)
        v2 = get_vertices(pos2[0], pos2[1], ang2)
        
        # Check if any vertex of 1 is in 2
        for p in v1:
            if point_in_hex(p, c2, ang2):
                return True
        # Check if any vertex of 2 is in 1
        for p in v2:
            if point_in_hex(p, c1, ang1):
                return True
        # Check edge intersections
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def calc_max_projection(verts_all):
        projs = verts_all @ normals.T
        return np.max(projs)

    def run_local_search(initial_pos, initial_angles, steps=10000):
        pos = np.array(initial_pos)
        angles = np.array(initial_angles)
        best_cost = float('inf')
        best_pos = pos.copy()
        best_angles = angles.copy()
        
        # Precompute verts for all
        verts_list = [get_vertices(pos[i,0], pos[i,1], angles[i]) for i in range(N)]
        
        def get_cost(p, a, verts_l):
            verts_all = np.vstack(verts_l)
            m = calc_max_projection(verts_all)
            return m

        current_cost = get_cost(pos, angles, verts_list)
        
        for step in range(steps):
            if time.time() > deadline:
                break
            idx = np.random.randint(N)
            delta_pos = np.random.normal(0, 0.05, 2)
            delta_ang = np.random.normal(0, 0.1)
            
            new_pos = pos.copy()
            new_angles = angles.copy()
            new_pos[idx] += delta_pos
            new_angles[idx] += delta_ang
            
            # Check overlap for idx against all others
            valid = True
            for j in range(N):
                if i == j: continue
                if check_overlap_full(new_pos[idx], new_angles[idx], pos[j], angles[j]):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Update verts for idx
            new_verts_list = verts_list.copy()
            new_verts_list[idx] = get_vertices(new_pos[idx,0], new_pos[idx,1], new_angles[idx])
            
            new_cost = get_cost(new_pos, new_angles, new_verts_list)
            
            if new_cost < current_cost:
                pos = new_pos
                angles = new_angles
                verts_list = new_verts_list
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
                    best_angles = angles.copy()
            else:
                T = 0.1 * (1 - step/steps)
                if T > 0 and np.random.rand() < math.exp(-(new_cost - current_cost) / T):
                    pos = new_pos
                    angles = new_angles
                    verts_list = new_verts_list
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_pos = pos.copy()
                        best_angles = angles.copy()
        
        return best_pos, best_angles, best_cost

    # Initial guess
    init_pos = []
    init_ang = []
    init_pos.append([0.0, 0.0])
    init_ang.append(0.0)
    d = math.sqrt(3)
    for k in range(6):
        x = d * math.cos(k * math.pi / 3)
        y = d * math.sin(k * math.pi / 3)
        init_pos.append([x, y])
        init_ang.append(0.0)
    for k in range(4):
        angle = np.random.uniform(0, 2*math.pi)
        dist = np.random.uniform(2.5, 3.5)
        init_pos.append([dist * math.cos(angle), dist * math.sin(angle)])
        init_ang.append(0.0)
        
    best_pos, best_angles, best_score = run_local_search(init_pos, init_ang, steps=20000)
    
    all_verts = [get_vertices(best_pos[i,0], best_pos[i,1], best_angles[i]) for i in range(N)]
    verts_all = np.vstack(all_verts)
    M = calc_max_projection(verts_all)
    S = 2 * M / math.sqrt(3)
    
    inner = []
    for i in range(N):
        inner.append([best_pos[i,0], best_pos[i,1], best_angles[i] * 180 / math.pi])
        
    return inner, (0.0, 0.0), S, 0.0

