"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        ang = a + math.radians(60 * k)
        verts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
    return np.array(verts)

def _outer_side_for_points(points, center, angle_deg):
    cx, cy = center
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        ang = a + math.radians(30 + 60 * k)
        normals.append((math.cos(ang), math.sin(ang)))
    normals = np.array(normals)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    proj = dx * normals[:, 0] + dy * normals[:, 1]
    proj = np.maximum(proj, 0.0)
    max_proj = proj.max()
    side = max_proj / (math.sqrt(3.0) / 2.0)
    return side

def _build_lattice_positions():
    e1 = np.array([math.sqrt(3.0), 0.0])
    e2 = np.array([math.sqrt(3.0)/2.0, 1.5])
    coords = []
    for i in range(-2, 2):
        for j in range(-1, 2):
            coords.append((i, j))
    pos = []
    for (i, j) in coords:
        p = i * e1 + j * e2
        pos.append(p)
    return np.array(pos)

def _inner_hexagons_from_positions(positions, angles):
    inner = []
    all_verts = []
    for k, (p, ang) in enumerate(zip(positions, angles)):
        inner.append([float(p[0]), float(p[1]), float(ang)])
        v = _hex_vertices(p[0], p[1], ang)
        all_verts.append(v)
    return inner, np.vstack(all_verts)

def construct():
    t0 = time.time()
    deadline = 150.0
    safety = 120.0

    positions = _build_lattice_positions()
    angles = np.zeros(N)

    inner, verts = _inner_hexagons_from_positions(positions, angles)
    center = (0.0, 0.0)
    angle = 0.0
    best_side = _outer_side_for_points(verts, center, angle)
    best_inner = [row[:] for row in inner]
    best_center = (0.0, 0.0)
    best_angle = 0.0

    def optimize_center(verts, angle_deg, init_center, iters=60):
        cx, cy = init_center
        best = _outer_side_for_points(verts, (cx, cy), angle_deg)
        step = 1.0
        for it in range(iters):
            improved = False
            for (dx, dy) in [(step,0),(-step,0),(0,step),(0,-step),
                             (step,step),(-step,step),(step,-step),(-step,-step)]:
                nc = (cx + dx, cy + dy)
                s = _outer_side_for_points(verts, nc, angle_deg)
                if s < best - 1e-12:
                    cx, cy = nc
                    best = s
                    improved = True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        return (cx, cy), best

    for ang in range(0, 60, 5):
        if time.time() > deadline - safety:
            break
        c, s = optimize_center(verts, float(ang), (0.0, 0.0), iters=40)
        if s < best_side - 1e-9:
            best_side = s
            best_center = c
            best_angle = float(ang)

    lo = best_angle - 5
    hi = best_angle + 5
    for _ in range(3):
        if time.time() > deadline - safety:
            break
        best_local = best_side
        for ang in np.linspace(lo, hi, 41):
            c, s = optimize_center(verts, float(ang), best_center, iters=30)
            if s < best_local - 1e-12:
                best_local = s
                best_center = c
                best_angle = float(ang)
        lo = best_angle - 2.5
        hi = best_angle + 2.5

    try:
        from scipy.optimize import minimize
        def objective(params):
            if time.time() > deadline - safety:
                return 1e9
            tx, ty = params[0], params[1]
            jit = params[2:2+2*N].reshape(N, 2)
            new_pos = positions + jit + np.array([tx, ty])
            _, v = _inner_hexagons_from_positions(new_pos, angles)
            c, s = optimize_center(v, best_angle, (0.0, 0.0), iters=20)
            return s
        x0 = np.zeros(2 + 2*N)
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-4,
                                'adaptive': True})
        if time.time() <= deadline - safety:
            if res.fun < best_side - 1e-9:
                tx, ty = res.x[0], res.x[1]
                jit = res.x[2:2+2*N].reshape(N, 2)
                new_pos = positions + jit + np.array([tx, ty])
                _, v = _inner_hexagons_from_positions(new_pos, angles)
                c, s = optimize_center(v, best_angle, (0.0, 0.0), iters=40)
                if s < best_side - 1e-9:
                    best_side = s
                    best_center = c
                    inner, _ = _inner_hexagons_from_positions(new_pos, angles)
                    best_inner = [row[:] for row in inner]
    except Exception:
        pass

    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END

