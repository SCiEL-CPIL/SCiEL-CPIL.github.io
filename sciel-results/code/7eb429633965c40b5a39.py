import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = time.time() + 130.0


def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def point_in_convex_polygon(px, py, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def polygons_overlap_sat(A, B, eps=1e-9):
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - eps or maxB < minA - eps:
                return False
    return True


def min_outer_side(inner, oc, oang):
    cx, cy = oc
    a = math.radians(oang)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj


def overlap_penalty(polys):
    penalty = 0.0
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            A, B = polys[i], polys[j]
            # Check overlap via SAT
            overlapping = False
            min_pen = float('inf')
            for poly in (A, B):
                m = len(poly)
                for k in range(m):
                    x1, y1 = poly[k]
                    x2, y2 = poly[(k + 1) % m]
                    dx, dy = x2 - x1, y2 - y1
                    length = math.sqrt(dx * dx + dy * dy)
                    if length < 1e-12:
                        continue
                    nx, ny = -dy / length, dx / length
                    minA = min(nx * px + ny * py for px, py in A)
                    maxA = max(nx * px + ny * py for px, py in A)
                    minB = min(nx * px + ny * py for px, py in B)
                    maxB = max(nx * px + ny * py for px, py in B)
                    if maxA < minB:
                        pass  # separated
                    elif maxB < minA:
                        pass  # separated
                    else:
                        overlapping = True
                        # penetration depth on this axis
                        pen = min(maxA, maxB) - max(minA, minB)
                        if pen < min_pen:
                            min_pen = pen
            if overlapping:
                if min_pen < float('inf'):
                    penalty += min_pen
    return penalty


def objective(params, oc=(0.0, 0.0), oang=0.0):
    inner = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(N)]
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    os = min_outer_side(inner, oc, oang)
    op = overlap_penalty(polys)
    return os + 50.0 * op


def build_initial(scale, oang):
    # hexagonal cluster: 1 center + 6 ring1 + 5 ring2
    pos = [(0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        pos.append((scale * math.cos(theta), scale * math.sin(theta)))
    for k in range(5):
        theta = k * math.pi / 3.0
        pos.append((2*scale * math.cos(theta), 2*scale * math.sin(theta)))
    return [(x, y, 0.0) for (x, y) in pos]


def is_valid(inner, oc, os, oang):
    outer = hex_vertices(oc[0], oc[1], os, oang)
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j], eps=1e-7):
                return False
    return True


def local_refine(x0, oc, oang, max_iter=300):
    def obj(x):
        return objective(x, oc, oang)
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((-90, 90))
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-7})
    return res


def run_de(oc, oang, popsize=15, maxiter=60, seed=42):
    bounds = []
    for i in range(N):
        bounds.append((-4, 4))
        bounds.append((-4, 4))
        bounds.append((-60, 60))
    try:
        res = differential_evolution(
            lambda p: objective(p, oc, oang),
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-4,
            mutation=(0.4, 1.2),
            recombination=0.9,
            seed=seed,
            polish=False,
            init='latinhypercube'
        )
        return res.x, res.fun
    except Exception:
        return None, float('inf')


def construct():
    t0 = time.time()
    best_side = 1e9
    best_inner = None
    best_oc = (0.0, 0.0)
    best_oang = 0.0

    # Precompute a few promising outer angles
    outer_angles = [0.0, 15.0, 30.0, 45.0, 60.0]

    for oang in outer_angles:
        if time.time() - t0 > 120:
            break

        # Build initial guess
        init_inner = build_initial(2.0, oang)
        init_params = np.array([c for tri in init_inner for c in tri])

        # DE search
        x_best, f_best = run_de((0.0, 0.0), oang, popsize=15, maxiter=50, seed=42)

        if x_best is not None:
            inner = [(x_best[i*3], x_best[i*3+1], x_best[i*3+2]) for i in range(N)]
            os = min_outer_side(inner, (0.0, 0.0), oang)
            if is_valid(inner, (0.0, 0.0), os, oang) and os < best_side:
                best_side = os
                best_inner = inner
                best_oc = (0.0, 0.0)
                best_oang = oang

        # Local refinement from DE result
        if x_best is not None and time.time() - t0 < 120:
            res = local_refine(x_best, (0.0, 0.0), oang, max_iter=200)
            inner = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(N)]
            os = min_outer_side(inner, (0.0, 0.0), oang)
            if is_valid(inner, (0.0, 0.0), os, oang) and os < best_side:
                best_side = os
                best_inner = inner
                best_oc = (0.0, 0.0)
                best_oang = oang

    # Fallback if nothing found
    if best_inner is None:
        init_inner = build_initial(2.0, 0.0)
        os = min_outer_side(init_inner, (0.0, 0.0), 0.0)
        return init_inner, (0.0, 0.0), os, 0.0

    return best_inner, best_oc, best_side, best_oang
