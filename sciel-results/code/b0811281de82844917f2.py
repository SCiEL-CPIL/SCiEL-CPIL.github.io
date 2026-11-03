import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 115.0

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    def get_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _UNIT_HEX_VERTS[0][0]*c - _UNIT_HEX_VERTS[0][1]*s # Placeholder, need full vectorization
        # Correct vectorization:
        # _UNIT_HEX_VERTS is (6, 2)
        # c, s are (N,)
        # We want (N, 6, 2)
        vx = _UNIT_HEX_VERTS[:, 0][None, :] * c[:, None] - _UNIT_HEX_VERTS[:, 1][None, :] * s[:, None]
        vy = _UNIT_HEX_VERTS[:, 0][None, :] * s[:, None] + _UNIT_HEX_VERTS[:, 1][None, :] * c[:, None]
        verts_x = positions[:, 0:1] + vx
        verts_y = positions[:, 1:2] + vy
        return verts_x, verts_y

    def compute_objective(positions, angles_rad):
        vx, vy = get_vertices(positions, angles_rad)
        flat_x = vx.flatten()
        flat_y = vy.flatten()
        
        mean_x = np.mean(flat_x)
        mean_y = np.mean(flat_y)
        
        off_x = flat_x - mean_x
        off_y = flat_y - mean_y
        
        # Project onto 6 normals
        # _EDGE_NORMALS (6, 2)
        # off_x (M,), off_y (M,)
        projs = off_x[None, :] * _EDGE_NORMALS[:, 0][:, None] + off_y[None, :] * _EDGE_NORMALS[:, 1][:, None]
        max_abs = np.max(np.abs(projs), axis=1)
        max_proj = np.max(max_abs)
        
        side = (2.0 / math.sqrt(3.0)) * max_proj
        return side, (float(mean_x), float(mean_y))

    def are_separated(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        if dx*dx + dy*dy > 4.5:
            return True
            
        c1, s1 = math.cos(h1[2]), math.sin(h1[2])
        c2, s2 = math.cos(h2[2]), math.sin(h2[2])
        
        v1x = h1[0] + _UNIT_HEX_VERTS[:, 0]*c1 - _UNIT_HEX_VERTS[:, 1]*s1
        v1y = h1[1] + _UNIT_HEX_VERTS[:, 0]*s1 + _UNIT_HEX_VERTS[:, 1]*c1
        v2x = h2[0] + _UNIT_HEX_VERTS[:, 0]*c2 - _UNIT_HEX_VERTS[:, 1]*s2
        v2y = h2[1] + _UNIT_HEX_VERTS[:, 0]*s2 + _UNIT_HEX_VERTS[:, 1]*c2
        
        en1x = _EDGE_NORMALS[:, 0]*c1 - _EDGE_NORMALS[:, 1]*s1
        en1y = _EDGE_NORMALS[:, 0]*s1 + _EDGE_NORMALS[:, 1]*c1
        en2x = _EDGE_NORMALS[:, 0]*c2 - _EDGE_NORMALS[:, 1]*s2
        en2y = _EDGE_NORMALS[:, 0]*s2 + _EDGE_NORMALS[:, 1]*c2
        
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x*nx + v1y*ny
            p2 = v2x*nx + v2y*ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x*nx + v1y*ny
            p2 = v2x*nx + v2y*ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def is_valid(positions, angles_rad):
        for i in range(N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            for j in range(i+1, N):
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if not are_separated(h1, h2):
                    return False
        return True

    def make_honeycomb():
        a1 = np.array([math.sqrt(3), 0.0])
        a2 = np.array([math.sqrt(3)/2.0, 1.5])
        coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0)
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i, (u, v) in enumerate(coords):
            positions[i] = u*a1 + v*a2
        return positions, angles

    def make_random():
        positions = np.random.uniform(-5, 5, (N, 2))
        angles = np.random.uniform(0, 2*math.pi, N)
        return positions, angles

    # Multi-start SA
    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_center = (0,0)
    
    inits = [make_honeycomb(), make_random(), make_random()]
    
    for pos0, ang0 in inits:
        if not is_valid(pos0, ang0):
            # Try to fix or skip
            continue
            
        side0, center0 = compute_objective(pos0, ang0)
        if side0 < best_side:
            best_side = side0
            best_pos = pos0.copy()
            best_ang = ang0.copy()
            best_center = center0

        # SA
        T = 0.2
        T_min = 0.001
        cooling = 0.9999
        step = 0.5
        pos = pos0.copy()
        ang = ang0.copy()
        cur_side, cur_center = compute_objective(pos, ang)
        
        start_time = time.time()
        while time.time() - start_time < 30.0: # 30s per start
            idx = random.randint(0, N-1)
            new_pos = pos.copy()
            new_ang = ang.copy()
            
            new_pos[idx, 0] += random.gauss(0, step)
            new_pos[idx, 1] += random.gauss(0, step)
            new_ang[idx] += random.gauss(0, 0.1)
            
            if is_valid(new_pos, new_ang):
                new_side, new_center = compute_objective(new_pos, new_ang)
                delta = new_side - cur_side
                if delta < 0 or random.random() < math.exp(-delta/T):
                    pos = new_pos
                    ang = new_ang
                    cur_side = new_side
                    cur_center = new_center
                    
                    if cur_side < best_side:
                        best_side = cur_side
                        best_pos = pos.copy()
                        best_ang = ang.copy()
                        best_center = cur_center
            T *= cooling
            if T < T_min: T = T_min
            step *= 0.9999
            if step < 0.01: step = 0.01

    # Polish
    if best_pos is not None:
        def obj_func(params):
            p = np.zeros((N, 2))
            a = np.zeros(N)
            for i in range(N):
                p[i, 0] = params[3*i]
                p[i, 1] = params[3*i+1]
                a[i] = params[3*i+2]
            if not is_valid(p, a):
                return 1e6
            s, _ = compute_objective(p, a)
            return s

        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = best_pos[i, 0]
            init_params[3*i+1] = best_pos[i, 1]
            init_params[3*i+2] = best_ang[i]

        try:
            res = minimize(obj_func, init_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_side:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                if is_valid(new_pos, new_ang):
                    best_side, best_center = compute_objective(new_pos, new_ang)
                    best_pos = new_pos
                    best_ang = new_ang
        except:
            pass

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(np.degrees(best_ang[i]))])
        
    return inner, best_center, float(best_side), 0.0

