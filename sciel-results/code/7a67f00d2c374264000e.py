"""Packing 12 unit hexagons inside a regular hexagon.

Strategy:
  1. Build a good initial placement (hexagonal grid / known patterns).
  2. Compute the minimal enclosing regular hexagon via convex projection.
  3. Use simulated annealing with a smooth surrogate (log-sum-exp) for the
     max objective, plus a final high-precision polish via Nelder-Mead.
  4. Also try varying the outer hexagon rotation.
"""

import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 135.0  # 15s margin before 150s kill


def _elapsed():
    return time.time() - START_TIME


def get_hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side, rotation."""
    angle_rad = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts[k, 0] = cx + side * np.cos(a)
        verts[k, 1] = cy + side * np.sin(a)
    return verts


def compute_minimal_enclosing_side(inner_hexes, outer_rotation_deg=0.0):
    """Compute minimal side length of outer hexagon (with given rotation)
    that contains all inner hexagons. Returns (side, center_x, center_y)."""
    all_verts = np.vstack([
        get_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner_hexes
    ])

    rot_rad = np.radians(outer_rotation_deg)
    normals = np.empty((6, 2))
    for k in range(6):
        a = rot_rad + k * np.pi / 3 + np.pi / 6
        normals[k, 0] = np.cos(a)
        normals[k, 1] = np.sin(a)

    proj_max = np.max(np.dot(all_verts, normals.T), axis=0)
    proj_min = np.min(np.dot(all_verts, normals.T), axis=0)

    def objective(c):
        cx, cy = c
        center_dot = cx * normals[:, 0] + cy * normals[:, 1]
        hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
        return hw.max()

    c0 = np.array([all_verts[:, 0].mean(), all_verts[:, 1].mean()])
    result = minimize(objective, c0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})

    cx_opt, cy_opt = result.x
    center_dot = cx_opt * normals[:, 0] + cy_opt * normals[:, 1]
    hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
    L = hw.max() / (np.sqrt(3) / 2)

    return L, cx_opt, cy_opt


def polygons_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    for verts in (v1, v2):
        for i in range(6):
            a = verts[i]
            b = verts[(i + 1) % 6]
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                return False
    return True


def check_no_overlap(inner_hexes, eps=-1e-5):
    """Check that no two inner hexagons overlap."""
    verts_list = [get_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner_hexes]
    for i in range(N):
        for j in range(i + 1, N):
            if polygons_overlap(verts_list[i], verts_list[j], eps=eps):
                return False
    return True


def construct():
    global START_TIME
    START_TIME = time.time()

    np.random.seed(42)

    # Initial configuration: hexagonal grid arrangement
    # For unit hexagons in a honeycomb:
    # Horizontal spacing: 3 (touching side-to-side)
    # Vertical spacing: sqrt(3)
    # Offset every other row by 1.5
    dx = 3.0
    dy = np.sqrt(3)

    inner = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx * 0.5
            y = row * dy
            inner.append([x, y, 0.0])

    # Compute initial enclosing side
    side, cx, cy = compute_minimal_enclosing_side(inner, 0.0)
    best_side = side
    best_inner = [list(h) for h in inner]
    best_center = (cx, cy)
    best_rot = 0.0

    # Simulated annealing
    T = 0.5
    T_min = 0.001
    cooling = 0.997
    max_iters = 8000

    current_inner = [list(h) for h in inner]
    current_side = side

    for it in range(max_iters):
        if _elapsed() > TIME_LIMIT:
            break

        idx = np.random.randint(N)
        new_inner = [list(h) for h in current_inner]
        step_size = 0.3 * (1.0 + T)
        new_inner[idx][0] += np.random.normal(0, step_size)
        new_inner[idx][1] += np.random.normal(0, step_size)
        new_inner[idx][2] = (new_inner[idx][2] + np.random.normal(0, 5.0)) % 60.0

        # Quick overlap check
        if not check_no_overlap(new_inner):
            T = max(T * cooling, T_min)
            continue

        new_side, new_cx, new_cy = compute_minimal_enclosing_side(new_inner, 0.0)

        delta = new_side - current_side
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_inner = new_inner
            current_side = new_side
            if current_side < best_side:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_center = (new_cx, new_cy)

        T = max(T * cooling, T_min)

    # Try different outer rotations
    for rot in [0.0, 30.0, 15.0, 45.0, 10.0, 20.0]:
        if _elapsed() > TIME_LIMIT:
            break
        side_r, cx_r, cy_r = compute_minimal_enclosing_side(best_inner, rot)
        if side_r < best_side:
            best_side = side_r
            best_center = (cx_r, cy_r)
            best_rot = rot

    # Final refinement: smaller steps, more iterations
    T = 0.02
    for it in range(4000):
        if _elapsed() > TIME_LIMIT:
            break
        idx = np.random.randint(N)
        new_inner = [list(h) for h in best_inner]
        step_size = 0.08
        new_inner[idx][0] += np.random.normal(0, step_size)
        new_inner[idx][1] += np.random.normal(0, step_size)
        new_inner[idx][2] = (new_inner[idx][2] + np.random.normal(0, 2.0)) % 60.0

        if not check_no_overlap(new_inner):
            T = max(T * 0.99, 0.001)
            continue

        new_side, new_cx, new_cy = compute_minimal_enclosing_side(new_inner, best_rot)

        delta = new_side - best_side
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            best_inner = new_inner
            best_side = new_side
            best_center = (new_cx, new_cy)

        T = max(T * 0.99, 0.001)

    # High-precision polish using Nelder-Mead on all parameters
    if _elapsed() < TIME_LIMIT - 10:
        # Flatten parameters: x, y, angle for each hexagon
        params = np.array([p for h in best_inner for p in h])

        def full_objective(p):
            if _elapsed() > TIME_LIMIT - 5:
                return best_side + 1.0
            hexes = [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)]
            if not check_no_overlap(hexes, eps=-1e-4):
                return best_side + 10.0
            s, _, _ = compute_minimal_enclosing_side(hexes, best_rot)
            return s

        result = minimize(full_objective, params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_side:
            best_inner = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
            best_side = result.fun
            best_side, best_center[0], best_center[1] = compute_minimal_enclosing_side(best_inner, best_rot)

    # Final precise computation
    final_side, final_cx, final_cy = compute_minimal_enclosing_side(best_inner, best_rot)

    return best_inner, (final_cx, final_cy), final_side, best_rot
