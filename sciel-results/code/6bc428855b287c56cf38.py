import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Constants
    HEX_SIDE = 1.0
    NORMALS = np.array([
        [math.cos(math.radians(30 + 60*i)), math.sin(math.radians(30 + 60*i))]
        for i in range(6)
    ])
    
    def get_vertices(cx, cy, angle_deg):
        ang = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = ang + i * math.pi / 3
            verts.append([cx + math.cos(a), cy + math.sin(a)])
        return np.array(verts)

    def point_in_hex(p, hex_center, hex_angle_deg):
        # Check if point p is inside hexagon centered at hex_center with angle hex_angle_deg
        # Transform p to hex local coords
        dx = p[0] - hex_center[0]
        dy = p[1] - hex_center[1]
        ang = math.radians(hex_angle_deg)
        # Rotate point by -ang
        cos_a = math.cos(-ang)
        sin_a = math.sin(-ang)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # Hexagon in local coords (vertices at 0, 60, ...)
        # Inequalities:
        # |lx| <= 1
        # |ly| <= sqrt(3)/2
        # |sqrt(3)*ly + lx| <= sqrt(3)  (Wait, need to check normals)
        # Normals in local coords are at 30, 90, ...
        # n0: (sqrt(3)/2, 1/2) -> lx*sqrt(3)/2 + ly/2 <= sqrt(3)/2
        # n1: (0, 1) -> ly <= sqrt(3)/2
        # n2: (-sqrt(3)/2, 1/2) -> -lx*sqrt(3)/2 + ly/2 <= sqrt(3)/2
        # etc.
        
        val1 = abs(lx)
        val2 = abs(ly)
        val3 = abs(math.sqrt(3)*ly + lx)
        val4 = abs(math.sqrt(3)*ly - lx)
        
        # Actually, simpler: check distance to edges
        # Or just use the 6 inequalities
        # d_k = lx * n_kx + ly * n_ky
        # n_k in local frame:
        # 30 deg: (sqrt(3)/2, 1/2)
        # 90 deg: (0, 1)
        # 150 deg: (-sqrt(3)/2, 1/2)
        # 210 deg: (-sqrt(3)/2, -1/2)
        # 270 deg: (0, -1)
        # 330 deg: (sqrt(3)/2, -1/2)
        
        d0 = lx * math.sqrt(3)/2 + ly * 0.5
        d1 = ly
        d2 = -lx * math.sqrt(3)/2 + ly * 0.5
        d3 = -lx * math.sqrt(3)/2 - ly * 0.5
        d4 = -ly
        d5 = lx * math.sqrt(3)/2 - ly * 0.5
        
        max_d = max(abs(d0), abs(d1), abs(d2), abs(d3), abs(d4), abs(d5))
        return max_d <= math.sqrt(3)/2 + 1e-9

    def min_distance_hex_hex(v1, v2):
        # v1, v2 are arrays of vertices (6, 2)
        min_d = float('inf')
        # Check vertex of v1 to edges of v2
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2_start = v2[j]
                p2_end = v2[(j+1)%6]
                # Distance from point p1 to segment p2_start-p2_end
                d_vec = p2_end - p2_start
                t = np.dot(p1 - p2_start, d_vec) / np.dot(d_vec, d_vec)
                t = max(0, min(1, t))
                proj = p2_start + t * d_vec
                dist = np.linalg.norm(p1 - proj)
                if dist < min_d:
                    min_d = dist
        # Check vertex of v2 to edges of v1
        for i in range(6):
            p2 = v2[i]
            for j in range(6):
                p1_start = v1[j]
                p1_end = v1[(j+1)%6]
                d_vec = p1_end - p1_start
                t = np.dot(p2 - p1_start, d_vec) / np.dot(d_vec, d_vec)
                t = max(0, min(1, t))
                proj = p1_start + t * d_vec
                dist = np.linalg.norm(p2 - proj)
                if dist < min_d:
                    min_d = dist
        return min_d

    def objective(params):
        # params: 12 * 3 = 36 floats
        # [x1, y1, a1, x2, y2, a2, ...]
        verts_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            v = get_vertices(x, y, a)
            verts_list.append(v)
        
        # Boundary violation
        max_proj = 0
        for v in verts_list:
            # Project all vertices on all normals
            projs = v @ NORMALS.T
            max_proj = max(max_proj, np.max(np.abs(projs)))
        
        S_req = 2.0 / math.sqrt(3) * max_proj
        
        # Overlap penalty
        overlap_pen = 0
        for i in range(N):
            for j in range(i+1, N):
                d = min_distance_hex_hex(verts_list[i], verts_list[j])
                if d < 1e-6:
                    overlap_pen += (1e-6 - d)**2
                else:
                    # We want d >= 0. If d < 0, they overlap.
                    # But min_distance is always >= 0.
                    # So if they are disjoint, d > 0.
                    # If they touch, d = 0.
                    # If they overlap, d = 0?
                    # No, min_distance between two overlapping convex sets is 0.
                    # So we need a different metric for overlap.
                    # Let's use the area of intersection? Too slow.
                    # Let's use the number of vertices of i inside j.
                    count_in = 0
                    for p in verts_list[i]:
                        if point_in_hex(p, [params[j*3], params[j*3+1]], params[j*3+2]):
                            count_in += 1
                    if count_in > 0:
                        overlap_pen += count_in**2
                    count_in = 0
                    for p in verts_list[j]:
                        if point_in_hex(p, [params[i*3], params[i*3+1]], params[i*3+2]):
                            count_in += 1
                    if count_in > 0:
                        overlap_pen += count_in**2

        return S_req + 100.0 * overlap_pen

    # Initial guesses
    def init_grid():
        # 4x3 grid
        params = []
        for i in range(4):
            for j in range(3):
                x = 2.0 * (i - 1.5)
                y = math.sqrt(3) * (j - 1.0)
                a = 0.0
                params.extend([x, y, a])
        return np.array(params)

    def init_honeycomb():
        # 7 hexes + 5 more
        params = []
        # Center
        params.extend([0, 0, 0])
        # 6 neighbors
        for k in range(6):
            ang = math.radians(60 * k)
            x = 2.0 * math.cos(ang)
            y = 2.0 * math.sin(ang)
            params.extend([x, y, 0])
        # 5 more: place them in the gaps?
        # Just randomize the last 5 slightly
        for k in range(5):
            ang = math.radians(60 * (k + 1)) + 30
            x = 3.5 * math.cos(ang)
            y = 3.5 * math.sin(ang)
            params.extend([x, y, 0])
        return np.array(params)

    best_score = float('inf')
    best_params = None
    
    inits = [init_grid(), init_honeycomb()]
    
    # Add random inits
    for _ in range(5):
        params = []
        for i in range(N):
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-4, 4)
            a = np.random.uniform(0, 180)
            params.extend([x, y, a])
        inits.append(np.array(params))

    for init in inits:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        best_params = init_grid()
        
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
    
    # Calculate final S
    verts_list = []
    for i in range(N):
        v = get_vertices(inner[i][0], inner[i][1], inner[i][2])
        verts_list.append(v)
        
    max_proj = 0
    for v in verts_list:
        projs = v @ NORMALS.T
        max_proj = max(max_proj, np.max(np.abs(projs)))
        
    S = 2.0 / math.sqrt(3) * max_proj
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
