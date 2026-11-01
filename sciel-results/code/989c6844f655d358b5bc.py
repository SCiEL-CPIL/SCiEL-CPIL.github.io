"""Packing 11 unit hexagons in a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon, center (cx,cy), side s, rotation angle_deg."""
    angle = np.radians(angle_deg)
    return np.array([[cx + s * np.cos(angle + i * np.pi / 3),
                      cy + s * np.sin(angle + i * np.pi / 3)] for i in range(6)])

def min_outer_side(positions, angles, ocx=0.0, ocy=0.0, oangle=0.0):
    """Min outer hexagon side to contain all unit hexagons at given positions/angles."""
    oang = np.radians(oangle)
    apothem_factor = np.sqrt(3) / 2
    max_proj = 0.0
    for i in range(N):
        verts = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for v in verts:
            dx, dy = v[0] - ocx, v[1] - ocy
            for j in range(6):
                na = oang + np.radians(30 + j * 60)
                proj = dx * np.cos(na) + dy * np.sin(na)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / apothem_factor

def objective(params):
    pos = [(params[3*i], params[3*i+1]) for i in range(N)]
    ang = [params[3*i+2] for i in range(N)]
    return min_outer_side(pos, ang, params[3*N], params[3*N+1], params[3*N+2])

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    s3 = np.sqrt(3)
    
    # Initial: honeycomb cluster
    pos = [(0.0, 0.0)]
    ang = [0.0]
    for i in range(6):
        a = i * np.pi / 3
        pos.append((s3 * np.cos(a), s3 * np.sin(a)))
        ang.append(0.0)
    for i in range(4):
        a = i * np.pi / 3
        pos.append((2*s3 * np.cos(a), 2*s3 * np.sin(a)))
        ang.append(0.0)
    
    x0 = []
    for i in range(N):
        x0 += [pos[i][0], pos[i][1], ang[i]]
    x0 += [0.0, 0.0, 0.0]
    
    best = x0[:]
    best_val = objective(x0)
    
    # Nelder-Mead
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
    if res.fun < best_val:
        best_val = res.fun
        best = res.x
    
    if time.time() > deadline - 5:
        p = [(best[3*i], best[3*i+1]) for i in range(N)]
        a = [best[3*i+2] for i in range(N)]
        s = min_outer_side(p, a, best[3*N], best[3*N+1], best[3*N+2])
        return [[p[i][0], p[i][1], a[i]] for i in range(N)], (best[3*N], best[3*N+1]), s, best[3*N+2]
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        x_r = x0 + rng.normal(0, 0.5, len(x0))
        res = minimize(objective, x_r, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_val:
            best_val = res.fun
            best = res.x
    
    p = [(best[3*i], best[3*i+1]) for i in range(N)]
    a = [best[3*i+2] for i in range(N)]
    s = min_outer_side(p, a, best[3*N], best[3*N+1], best[3*N+2])
    return [[p[i][0], p[i][1], a[i]] for i in range(N)], (best[3*N], best[3*N+1]), s, best[3*N+2]
