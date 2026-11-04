import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 145.0

def _hex_vertices(cx, cy, s, rot_deg):
    a0 = math.radians(rot_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex_pts(cx, cy, ang_deg):
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot_deg):
    theta_rad = math.radians(rot_deg)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    max_proj = 0.0
    for (x, y, a) in inner:
        pts = _inner_hex_pts(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / math.sqrt(3.0)

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_poly(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i+1) % n], pt)
        if abs(c) < 1e-9:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def _poly_overlap(A, B):
    """SAT: return True if convex polygons A and B overlap (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            if maxA < minB - 1e-7 or maxB < minA - 1e-7:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex_pts(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _hexagonal_seed():
    """
    Create a hexagonal lattice seed: 1 center + 6 ring + 4 from second ring
    """
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0) * 1.001
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _hexagonal_seed2():
    """Alternative: 1 center + 6 ring + 4 from second ring at different angles"""
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0) * 1.001
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(90 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _hexagonal_seed3():
    """Another variant with slight rotation"""
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0) * 1.001
    for k in range(6):
        a = math.radians(0 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(0 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _objective(params):
    """
    Objective: minimize outer side length with penalties for overlaps and boundary violations.
    params: [x1,y1,a1, ..., x11,y11,a11, ox, oy, rot]
    """
    pts = params[:N*3].reshape(N, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    thetas = pts[:, 2]
    ox, oy = params[N*3], params[N*3+1]
    rot = params[N*3+2]
    
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    # Compute outer side length
    max_proj = 0.0
    for k in range(N):
        a0 = math.radians(thetas[k])
        for m in range(6):
            a = a0 + math.radians(60 * m)
            px = xs[k] + math.cos(a)
            py = ys[k] + math.sin(a)
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    side = 2.0 * max_proj / math.sqrt(3.0)
    
    # Penalties
    penalty = 0.0
    
    # Overlap penalty using proper polygon check (approximate with vertex checks)
    for i in range(N):
        pts_i = []
        a0 = math.radians(thetas[i])
        for m in range(6):
            a = a0 + math.radians(60 * m)
            pts_i.append((xs[i] + math.cos(a), ys[i] + math.sin(a)))
        
        for j in range(i+1, N):
            pts_j = []
            a0 = math.radians(thetas[j])
            for m in range(6):
                a = a0 + math.radians(60 * m)
                pts_j.append((xs[j] + math.cos(a), ys[j] + math.sin(a)))
            
            # Check if polygons overlap using SAT
            if _poly_overlap(pts_i, pts_j):
                # Compute minimum translation distance as penalty
                # Approximate: use center distance
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = math.sqrt(3.0)
                if dist < min_dist:
                    diff = min_dist - dist
                    penalty += diff * diff * 5000.0
    
    # Boundary violation penalty
    outer_s = side
    for k in range(N):
        a0 = math.radians(thetas[k])
        for m in range(6):
            a = a0 + math.radians(60 * m)
            px = xs[k] + math.cos(a)
            py = ys[k] + math.sin(a)
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                limit = outer_s * math.sqrt(3.0) / 2.0
                if proj > limit:
                    penalty += (proj - limit) ** 2 * 5000.0
    
    return side + penalty

def _refine(inner, oc, rot, rng, budget):
    init_params = np.array([p for (x, y, a) in inner for p in [x, y, a]] + [oc[0], oc[1], rot])
    
    try:
        res_nm = minimize(
            _objective,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
        )
        if res_nm.fun < float('inf'):
            params = res_nm.x
            pts = params[:N*3].reshape(N, 3)
            inner_nm = [[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])] for i in range(N)]
            oc_nm = (float(params[N*3]), float(params[N*3+1]))
            rot_nm = float(params[N*3+2])
            s_nm = _min_outer_side(inner_nm, oc_nm, rot_nm)
            
            if _valid(inner_nm, oc_nm, s_nm, rot_nm):
                return inner_nm, oc_nm, s_nm, rot_nm
    except Exception:
        pass
    
    return None

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 8.0
    rng = random.Random(42)
    np.random.seed(42)
    
    # Try multiple seeds
    seeds = [_hexagonal_seed(), _hexagonal_seed2(), _hexagonal_seed3()]
    
    for seed_inner, seed_oc, seed_s, seed_rot in seeds:
        seed_s_tight = _min_outer_side(seed_inner, seed_oc, seed_rot)
        if _valid(seed_inner, seed_oc, seed_s_tight, seed_rot):
            if seed_s_tight < best_score:
                best_score = seed_s_tight
                best = (seed_inner, seed_oc, seed_s_tight, seed_rot)
    
    # Refine best seed
    if best is not None and time.time() < budget - 10:
        inner_b, oc_b, s_b, rot_b = best
        refined = _refine(inner_b, oc_b, rot_b, rng, budget - 5)
        if refined:
            inner_r, oc_r, s_r, rot_r = refined
            if s_r < best_score:
                best_score = s_r
                best = (inner_r, oc_r, s_r, rot_r)
    
    # Perturbation restarts from best
    if best is not None:
        for trial in range(20):
            if time.time() > budget - 8:
                break
            
            perturbed_inner = []
            for (x, y, a) in best[0]:
                perturbed_inner.append([x + rng.gauss(0, 0.15), y + rng.gauss(0, 0.15), a + rng.gauss(0, 8)])
            perturbed_oc = (best[1][0] + rng.gauss(0, 0.15), best[1][1] + rng.gauss(0, 0.15))
            perturbed_rot = best[3] + rng.gauss(0, 8)
            
            refined = _refine(perturbed_inner, perturbed_oc, perturbed_rot, rng, budget - 5)
            if refined:
                inner_r, oc_r, s_r, rot_r = refined
                if s_r < best_score:
                    best_score = s_r
                    best = (inner_r, oc_r, s_r, rot_r)
    
    # Final refinement
    if best is not None and time.time() < budget - 3:
        inner_b, oc_b, s_b, rot_b = best
        refined = _refine(inner_b, oc_b, rot_b, rng, budget - 2)
        if refined:
            inner_r, oc_r, s_r, rot_r = refined
            if s_r < best_score:
                best_score = s_r
                best = (inner_r, oc_r, s_r, rot_r)
    
    if best is None:
        inner, oc, s, rot = _hexagonal_seed()
        s_tight = _min_outer_side(inner, oc, rot)
        return inner, (oc[0], oc[1]), s_tight, rot
    
    inner, oc, s, rot = best
    s_tight = _min_outer_side(inner, oc, rot)
    if s_tight < s and _valid(inner, oc, s_tight, rot):
        s = s_tight
    
    return inner, (oc[0], oc[1]), s, rot
