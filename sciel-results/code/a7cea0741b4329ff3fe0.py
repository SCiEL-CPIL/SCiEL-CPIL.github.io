"""Pack 12 unit hexagons in a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + a
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _compute_required_side(inner_pos, inner_ang, outer_center, outer_angle):
    normals = np.array([[np.cos(np.radians(60*k + outer_angle)),
                         np.sin(np.radians(60*k + outer_angle))] for k in range(6)])
    max_dot = 0.0
    for i in range(N):
        verts = _hex_vertices(inner_pos[i, 0], inner_pos[i, 1], 1.0, inner_ang[i])
        for v in verts:
            dots = (v - outer_center) @ normals.T
            max_dot = max(max_dot, np.max(dots))
    if max_dot < 0:
        return 0.0
    return 2.0 * max_dot / np.sqrt(3)

def _objective(params):
    ip = params[0:36].reshape(N, 3)
    ox, oy, oang = params[36], params[37], params[38]
    side = _compute_required_side(ip[:, :2], ip[:, 2], np.array([ox, oy]), oang)
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(ip[i, :2] - ip[j, :2])
            md = np.sqrt(3) - 0.01
            if d < md:
                pen += (md - d) ** 2
    return side + 10.0 * pen

def _initial_guess():
    np.random.seed(42)
    dy, dx = np.sqrt(3), 2.0
    pos = []
    for k in range(3):
        pos.append([k*dx - dx, -dy, 0.0])
    for k in range(4):
        pos.append([k*dx - dx - dx/2, 0.0, 0.0])
    for k in range(3):
        pos.append([k*dx - dx, dy, 0.0])
    p = np.array(pos) + 0.1 * np.random.randn(N, 3)
    params = np.zeros(39)
    params[0:36] = p.flatten()
    return params

def construct():
    deadline = time.time() + 140.0
    best_params = _initial_guess()
    best_score = _objective(best_params)
    
    T, T_min, cooling = 0.5, 0.001, 0.995
    cur = best_params.copy()
    cur_score = best_score
    
    for it in range(200000):
        if time.time() > deadline - 10.0 or T < T_min:
            break
        new = cur + 0.1 * T * np.random.randn(39)
        new[2::3] = new[2::3] % 60
        ns = _objective(new)
        if ns < cur_score:
            cur, cur_score = new, ns
            if ns < best_score:
                best_params, best_score = new.copy(), ns
        else:
            if np.random.rand() < np.exp(-(ns - cur_score) / T):
                cur, cur_score = new, ns
        T *= cooling
    
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_params, best_score = res.x, res.fun
        except:
            pass
    
    ip = best_params[0:36].reshape(N, 3)
    inner = [[float(ip[i,0]), float(ip[i,1]), float(ip[i,2])] for i in range(N)]
    center = (float(best_params[36]), float(best_params[37]))
    side = float(_compute_required_side(ip[:, :2], ip[:, 2], np.array(center), float(best_params[38])))
    angle = float(best_params[38])
    return inner, center, max(side, 2.0), angle
# EVOLVE-BLOCK-END
