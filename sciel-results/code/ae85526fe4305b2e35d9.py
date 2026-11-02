import math
import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
N = 11
APOTHEM = math.sqrt(3) / 2.0


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    from scipy.optimize import linprog, minimize_scalar

    def hex_vertices(cx, cy, r, theta):
        verts = np.array([[0.0, 0.0]] * 6)
        for i in range(6):
            ang = theta + i * math.pi / 3.0
            verts[i] = [cx + r * math.cos(ang), cy + r * math.sin(ang)]
        return verts

    def all_vertices(positions):
        verts_list = []
        for (x, y, theta) in positions:
            verts_list.extend(hex_vertices(x, y, 1.0, theta).tolist())
        return np.array(verts_list)

    def compute_enclosing(positions, outer_rotation):
        all_verts = all_vertices(positions)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for k in range(6):
            ang = outer_rotation + k * math.pi / 3.0 + math.pi / 6.0
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in all_verts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                max_proj = res.x[2]
                S = 2.0 * max_proj / APOTHEM
                return S, (res.x[0], res.x[1]), max_proj
            else:
                return float('inf'), None, float('inf')
        except Exception:
            return float('inf'), None, float('inf')

    def axial_to_pts(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y, 0.0))
        return pts

    candidates_axial = [
        # Honeycomb
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)],
        # 4x3 rhombus
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)],
        # Diagonal
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
        # Compact
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2)],
        # Star
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2)],
        # T
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (1, -1)],
        # Wide
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1), (3, 1), (0, 1), (1, -1), (2, -1)],
        # Plus shape
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)],
    ]

    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    best_positions = None

    for axial in candidates_axial:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        positions = axial_to_pts(axial)

        for beta in np.linspace(0, math.pi / 6, 7):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S, center, _ = compute_enclosing(positions, float(beta))
            if S < best_S:
                best_S = S
                best_beta = float(beta)
                best_center = center
                best_positions = positions

        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                res = minimize_scalar(
                    lambda b: compute_enclosing(positions, float(b))[0],
                    bounds=(0, math.pi / 6), method='bounded',
                    options={'maxiter': 100, 'xatol': 1e-10}
                )
                if res.fun < best_S:
                    S2, center2, _ = compute_enclosing(positions, float(res.x))
                    best_S = res.fun
                    best_beta = float(res.x)
                    best_center = center2
                    best_positions = positions
            except Exception:
                pass

    if best_center is None or best_S == float('inf'):
        positions = axial_to_pts(candidates_axial[0])
        S, best_center, _ = compute_enclosing(positions, 0.0)
        if best_center is None:
            best_center = (0.0, 0.0)
            best_S = 3.0
        best_beta = 0.0
        best_positions = positions

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    inner = [[float(x), float(y), float(t)] for (x, y, t) in best_positions]
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
