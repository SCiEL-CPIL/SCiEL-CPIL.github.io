import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_rad):
    verts = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def required_outer_side(all_vertices, outer_cx, outer_cy, outer_angle_rad):
    min_side = 0.0
    for (x, y) in all_vertices:
        dx = x - outer_cx
        dy = y - outer_cy
        rx = dx * np.cos(outer_angle_rad) + dy * np.sin(outer_angle_rad)
        ry = -dx * np.sin(outer_angle_rad) + dy * np.cos(outer_angle_rad)
        req1 = abs(rx) + abs(ry) / np.sqrt(3)
        req2 = 2 * abs(ry) / np.sqrt(3)
        min_side = max(min_side, req1, req2)
    return min_side

def make_cluster(arrangement, theta, tx, ty):
    positions = []
    for (bx, by) in arrangement:
        rx = bx * np.cos(theta) - by * np.sin(theta)
        ry = bx * np.sin(theta) + by * np.cos(theta)
        positions.append((rx + tx, ry + ty))
    return positions

def objective(params, arrangement):
    theta, tx, ty = params
    positions = make_cluster(arrangement, theta, tx, ty)
    all_vertices = []
    for (cx, cy) in positions:
        verts = hexagon_vertices(cx, cy, 1.0, theta)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    return required_outer_side(all_vertices, 0.0, 0.0, 0.0)

def center_arrangement(arr):
    cx = np.mean([p[0] for p in arr])
    cy = np.mean([p[1] for p in arr])
    return [(x - cx, y - cy) for (x, y) in arr]

def try_arrangement(arrangement, timeout=8):
    start = time.time()
    best_val = float('inf')
    best = None
    n_starts = 40
    for i in range(n_starts):
        if time.time() - start > timeout:
            break
        if i == 0:
            x0 = [0.0, 0.0, 0.0]
        else:
            x0 = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-3, 3), np.random.uniform(-3, 3)]
        res = minimize(objective, x0, args=(arrangement,), method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
        if res.fun < best_val:
            best_val = res.fun
            best = res.x
    return best_val, best

def construct():
    start = time.time()
    deadline = start + 140.0
    sqrt3 = np.sqrt(3)

    arr1 = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (sqrt3 / 2, 1.5),
        (-sqrt3 / 2, 1.5),
        (-sqrt3, 0.0),
        (-sqrt3 / 2, -1.5),
        (sqrt3 / 2, -1.5),
        (2 * sqrt3, 0.0),
        (0.0, 3.0),
        (-2 * sqrt3, 0.0),
        (0.0, -3.0),
    ]

    arr2 = []
    for i in range(4):
        arr2.append((i * sqrt3 - 1.5 * sqrt3, 0.0))
    for i in range(3):
        arr2.append((i * sqrt3 - sqrt3, 1.5))
    for i in range(4):
        arr2.append((i * sqrt3 - 1.5 * sqrt3, 3.0))

    arr3 = []
    for i in range(3):
        arr3.append((i * sqrt3 - sqrt3, 0.0))
    for i in range(5):
        arr3.append((i * sqrt3 - 2 * sqrt3, 1.5))
    for i in range(3):
        arr3.append((i * sqrt3 - sqrt3, 3.0))

    arr4 = []
    for i in range(2):
        arr4.append((i * sqrt3 - sqrt3 / 2, 0.0))
    for i in range(4):
        arr4.append((i * sqrt3 - 1.5 * sqrt3, 1.5))
    for i in range(4):
        arr4.append((i * sqrt3 - 1.5 * sqrt3, 3.0))
    arr4.append((0.0, 4.5))

    arr5 = []
    for i in range(5):
        arr5.append((i * sqrt3 - 2 * sqrt3, 0.0))
    for i in range(6):
        arr5.append((i * sqrt3 - 2.5 * sqrt3, 1.5))

    arrangements = [center_arrangement(a) for a in [arr1, arr2, arr3, arr4, arr5]]

    best_side = float('inf')
    best_params = None
    best_arr = None

    for arr in arrangements:
        if time.time() > deadline:
            break
        val, params = try_arrangement(arr, timeout=8)
        if val < best_side:
            best_side = val
            best_params = params
            best_arr = arr

    if best_params is not None:
        theta, tx, ty = best_params
        positions = make_cluster(best_arr, theta, tx, ty)
        inner = [[x, y, np.degrees(theta) % 360] for (x, y) in positions]
        return inner, (0.0, 0.0), best_side, 0.0
    else:
        inner = [[sqrt3 * i, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
