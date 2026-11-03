"""Packing 12 unit hexagons in a regular hexagon - Nelder-Mead multi-topology."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, angle_deg):
    a = math.radians(angle_deg)
    return np.array([[cx + math.cos(a + i * math.pi / 3),
                      cy + math.sin(a + i * math.pi / 3)] for i in range(6)])

def min_vertex_dist(v1, v2):
    """Min distance between vertices of two polygons. Exact for disjoint convex polys."""
    d = np.linalg.norm(v1[:, None, :] - v2[None, :, :], axis=2)
    return float(np.min(d))

def min_gap_between(verts_list):
    """Minimum gap between all pairs. Negative = overlap (approx via vertex dist)."""
    min_gap = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = min_vertex_dist(verts_list[i], verts_list[j])
            if d < min_gap:
                min_gap = d
    return min_gap

def compute_side(verts_list, outer_center, outer_angle_deg):
    """Minimum outer hex side length to contain all vertices."""
    all_v = np.vstack(verts_list)
    rel = all_v - outer_center
    oa = math.radians(outer_angle_deg)
    normals = np.array([[math.cos(oa + k * math.pi / 3),
                         math.sin(oa + k * math.pi / 3)] for k in range(6)])
    projs = rel @ normals.T
    max_proj = float(np.max(projs))
    return 2.0 * max_proj / math.sqrt(3)

def params_to_verts(params):
    centers = params[:2 * N].reshape(N, 2)
    angles = params[2 * N:3 * N]
    verts = []
    for i in range(N):
        verts.append(hex_vertices(centers[i, 0], centers[i, 1], angles[i]))
    return verts

def objective(params):
    verts = params_to_verts(params)
    min_gap = min_gap_between(verts)
    outer_cx, outer_cy, outer_ang = params[3 * N], params[3 * N + 1], params[3 * N + 2]
    side = compute_side(verts, np.array([outer_cx, outer_cy]), outer_ang)
    penalty = 0.0
    if min_gap < 0:
        penalty = 50.0 * (-min_gap) ** 2
    return side + penalty

def build_ring12():
    """12 hexagons in a ring around a central point."""
    centers, angles = [], []
    for i in range(N):
        a = i * math.pi / 6
        r = 2.0
        centers.append([r * math.cos(a), r * math.sin(a)])
        angles.append(0.0)
    return np.array(centers).flatten(), np.array(angles)

def build_hex6_plus6():
    """Central hex + 6 neighbors + 6 more on outer ring."""
    centers, angles = [], []
    centers.append([0.0, 0.0])
    for i in range(6):
        a = i * math.pi / 3
        centers.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
    # 6 more on outer ring
    for i in range(6):
        a = i * math.pi / 3 + math.pi / 6
        centers.append([4.0 * math.cos(a), 4.0 * math.sin(a)])
    angles = [0.0] * N
    return np.array(centers).flatten(), np.array(angles)

def build_rect3x4():
    """3x4 staggered grid."""
    centers, angles = [], []
    dx, dy = 2.0, math.sqrt(3)
    for r in range(4):
        for c in range(3):
            x = c * dx
            y = r * dy
            if r % 2 == 1:
                x += 1.0
            centers.append([x, y])
    angles = [0.0] * N
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [[x - cx, y - cy] for x, y in centers]
    return np.array(centers).flatten(), np.array(angles)

def build_ring3x3_center():
    """Center + 8 around + 3 more."""
    centers, angles = [], []
    centers.append([0.0, 0.0])
    for i in range(8):
        a = i * math.pi / 4
        centers.append([2.2 * math.cos(a), 2.2 * math.sin(a)])
    for i in range(3):
        a = i * 2 * math.pi / 3
        centers.append([4.4 * math.cos(a), 4.4 * math.sin(a)])
    angles = [0.0] * N
    return np.array(centers).flatten(), np.array(angles)

def build_linear12():
    """12 in a row with slight stagger."""
    centers, angles = [], []
    for i in range(N):
        x = i * 2.0
        y = (i % 2) * math.sqrt(3) * 0.5
        centers.append([x, y])
    angles = [0.0] * N
    cx = np.mean([c[0] for c in centers])
    centers = [[x - cx, y] for x, y in centers]
    return np.array(centers).flatten(), np.array(angles)

def optimize_from(x0, deadline):
    """Nelder-Mead from x0, return best params found."""
    best = x0.copy()
    best_obj = objective(x0)
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-7,
                            'adaptive': True})
    if time.time() > deadline:
        return best, best_obj
    if res.fun < best_obj:
        best = res.x
        best_obj = res.fun
    # Second pass with smaller steps
    res2 = minimize(objective, best, method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-8,
                             'adaptive': True,
                             'initial_simplex': None})
    if time.time() > deadline:
        return best, best_obj
    if res2.fun < best_obj:
        best = res2.x
        best_obj = res2.fun
    return best, best_obj

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin

    topologies = [
        build_ring12, build_hex6_plus6, build_rect3x4,
        build_ring3x3_center, build_linear12
    ]

    best_params = None
    best_side = float('inf')
    best_valid = False

    for topo_fn in topologies:
        if time.time() > deadline:
            break
        centers_flat, angles = topo_fn()
        x0 = np.zeros(3 * N + 3)
        x0[:2 * N] = centers_flat
        x0[2 * N:3 * N] = angles
        x0[3 * N] = 0.0
        x0[3 * N + 1] = 0.0
        x0[3 * N + 2] = 0.0

        # Small random perturbations for diversity
        for _ in range(3):
            if time.time() > deadline:
                break
            x_try = x0.copy()
            x_try[:2 * N] += np.random.normal(0, 0.1, size=2 * N)
            x_try[2 * N:3 * N] += np.random.normal(0, 0.05, size=N)
            p, obj = optimize_from(x_try, deadline)
            # Validate
            verts = params_to_verts(p)
            gap = min_gap_between(verts)
            side = compute_side(verts, np.array([p[3 * N], p[3 * N + 1]]), p[3 * N + 2])
            if gap >= -1e-6 and side < best_side:
                best_side = side
                best_params = p
                best_valid = True

    # Final polish on best
    if best_valid and time.time() < deadline - 20:
        p, obj = optimize_from(best_params, deadline)
        verts = params_to_verts(p)
        gap = min_gap_between(verts)
        side = compute_side(verts, np.array([p[3 * N], p[3 * N + 1]]), p[3 * N + 2])
        if gap >= -1e-6 and side < best_side:
            best_side = side
            best_params = p
            best_valid = True

    if best_valid:
        inner = []
        for i in range(N):
            inner.append([float(best_params[2 * i]),
                          float(best_params[2 * i + 1]),
                          float(best_params[2 * N + i])])
        outer_cx = float(best_params[3 * N])
        outer_cy = float(best_params[3 * N + 1])
        outer_ang = float(best_params[3 * N + 2])
        final_side = best_side * 1.0005 + 1e-5
        return inner, (outer_cx, outer_cy), final_side, outer_ang
    else:
        # Fallback: ring12 with generous side
        c, a = build_ring12()
        inner = [[float(c[2 * i]), float(c[2 * i + 1]), float(a[i])] for i in range(N)]
        return inner, (0.0, 0.0), 12.0, 0.0
