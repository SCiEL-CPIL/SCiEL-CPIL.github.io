import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.radians(np.array([0, 60, 120, 180, 240, 300]))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def sat_overlap(v1, v2):
    n1 = len(v1)
    n2 = len(v2)
    def project(vertices, axis):
        proj = vertices @ axis
        return proj.min(), proj.max()
    for i in range(n1):
        edge = v1[(i+1)%n1] - v1[i]
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis /= norm
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
    for i in range(n2):
        edge = v2[(i+1)%n2] - v2[i]
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis /= norm
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
    return True

def objective(params):
    inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
    ocx, ocy, oangle = params[3*N], params[3*N+1], params[3*N+2]
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if sat_overlap(v1, v2):
                return 1e6
    max_proj = 0.0
    for k in range(6):
        theta = np.radians(oangle + 30 + 60 * k)
        normal = np.array([np.cos(theta), np.sin(theta)])
        for (cx, cy, a) in inner:
            verts = hexagon_vertices(cx, cy, 1.0, a) - np.array([ocx, ocy])
            proj = verts @ normal
            max_proj = max(max_proj, proj.max(), -proj.min())
    return max_proj * 2.0 / np.sqrt(3)

def construct():
    t0 = time.time()
    deadline = t0 + 140
    configs = []
    c1 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        a = np.radians(60 * k)
        c1.append((2.0 * np.cos(a), 2.0 * np.sin(a), 0.0))
    for k in [1, 3, 5, 0]:
        a = np.radians(60 * k + 30)
        c1.append((2.0 * np.sqrt(3) * np.cos(a), 2.0 * np.sqrt(3) * np.sin(a), 0.0))
    configs.append(c1)
    c2 = []
    for i in range(3):
        c2.append((2.0 * i - 2.0, 0.0, 0.0))
    for i in range(4):
        c2.append((2.0 * i - 3.0, 2.0 * np.sqrt(3), 0.0))
    for i in range(3):
        c2.append((2.0 * i - 2.0, 4.0 * np.sqrt(3), 0.0))
    configs.append(c2)
    best_params, best_score = None, float('inf')
    np.random.seed(42)
    for config in configs:
        x0 = np.array([v for (x, y, a) in config for v in (x, y, a)] + [0.0, 0.0, 0.0])
        for r in range(10):
            if time.time() > deadline:
                break
            params = x0 if r == 0 else x0 + np.random.randn(len(x0)) * 0.5
            try:
                res = minimize(objective, params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3})
                if res.fun < best_score:
                    best_score, best_params = res.fun, res.x
            except Exception:
                pass
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    if best_params is None:
        best_params = np.zeros(3 * N + 3)
        best_score = objective(best_params)
    inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
    return inner, (best_params[3*N], best_params[3*N+1]), best_score, best_params[3*N+2]
