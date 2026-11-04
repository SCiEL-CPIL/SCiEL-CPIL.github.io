import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 150.0

def _hex_vertices(cx, cy, s, rot):
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    ox, oy = outer_c
    max_proj = 0.0
    for (x, y, a) in inner:
        pts = _inner_hex(x, y, a)
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
            if maxA + 1e-9 < minB or maxB + 1e-9 < minA:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _hexagonal_seed():
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _hexagonal_seed2():
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in [0, 1, 3, 4]:
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _hexagonal_seed3():
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in [0, 2, 4, 5]:
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _make_seed(rng):
    inner = [[rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 60)] for _ in range(N)]
    return inner, (0.0, 0.0), 10.0, 0.0

def _objective_cluster(params, rot, oc):
    """
    params: array of shape (N, 2) with [x, y] for each inner hex (all share angle=rot).
    """
    xs = params[:, 0]
    ys = params[:, 1]
    
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = oc
    max_proj = 0.0
    for k in range(N):
        a0 = math.radians(rot)
        for m in range(6):
            a = a0 + math.radians(60 * m)
            px = xs[k] + math.cos(a)
            py = ys[k] + math.sin(a)
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    side = 2.0 * max_proj / math.sqrt(3.0)
    
    # Rigorous overlap check
    penalty = 0.0
    polys = []
    for k in range(N):
        polys.append(_inner_hex(xs[k], ys[k], rot))
        
    for i in range(N):
        for j in range(i+1, N):
            if _poly_overlap(polys[i], polys[j]):
                # Calculate a penalty based on how much they overlap
                # Simple penalty: 1000.0 if overlapping
                penalty += 1000.0
    
    return side + penalty

def _objective_full(params, rot, oc):
    """
    params: array of shape (N, 3) with [x, y, angle_deg] for each inner hex.
    """
    pts = params.reshape(N, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    thetas = pts[:, 2]
    
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = oc
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
    
    # Rigorous overlap check
    penalty = 0.0
    polys = []
    for k in range(N):
        polys.append(_inner_hex(xs[k], ys[k], thetas[k]))
        
    for i in range(N):
        for j in range(i+1, N):
            if _poly_overlap(polys[i], polys[j]):
                penalty += 1000.0
    
    return side + penalty

def _scipy_optimize_cluster(init_xy, rot, oc, budget, maxiter=500):
    try:
        res = minimize(
            _objective_cluster,
            init_xy,
            args=(rot, oc),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7}
        )
        if res.fun < float('inf') and time.time() < budget:
            return res.x
    except Exception:
        pass
    return None

def _scipy_optimize_full(init_params, rot, oc, budget, maxiter=500):
    try:
        res = minimize(
            _objective_full,
            init_params,
            args=(rot, oc),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7}
        )
        if res.fun < float('inf') and time.time() < budget:
            return res.x
    except Exception:
        pass
    return None

def _try_refine(inner, outer_c, s, rot, rng, iters, budget):
    best = (inner, outer_c, s, rot)
    cur = (inner, outer_c, s, rot)
    
    s_tight = _min_outer_side(cur[0], cur[1], cur[3])
    if s_tight < cur[2]:
        cur = (cur[0], cur[1], s_tight, cur[3])
        if _valid(cur[0], cur[1], cur[2], cur[3]):
            best = cur
    
    it = 0
    while it < iters:
        if time.time() > budget:
            break
        it += 1
        cand = [[x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), a + rng.uniform(-15, 15)] for (x, y, a) in cur[0]]
        cand_oc = (cur[1][0] + rng.uniform(-0.2, 0.2), cur[1][1] + rng.uniform(-0.2, 0.2))
        cand_rot = cur[3] + rng.uniform(-10, 10)
        
        cand_s = _min_outer_side(cand, cand_oc, cand_rot)
        
        if _valid(cand, cand_oc, cand_s, cand_rot):
            cur = (cand, cand_oc, cand_s, cand_rot)
            if cand_s < best[2]:
                best = cur
    
    return best

def _anneal_cluster(inner, oc, rot, rng, budget):
    """
    Simulated annealing on the cluster positions (x,y) with fixed rotation.
    """
    xy = np.array([[p[0], p[1]] for p in inner])
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    best_xy = xy.copy()
    best_val = _objective_cluster(xy, rot, oc)
    
    it = 0
    while T > T_min and time.time() < budget:
        it += 1
        if it % 100 == 0 and time.time() > budget:
            break
        
        dx = rng.gauss(0, 0.2)
        dy = rng.gauss(0, 0.2)
        idx = rng.randint(0, N-1)
        
        new_xy = xy.copy()
        new_xy[idx, 0] += dx
        new_xy[idx, 1] += dy
        
        new_val = _objective_cluster(new_xy, rot, oc)
        delta = new_val - best_val
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            xy = new_xy
            if new_val < best_val:
                best_val = new_val
                best_xy = new_xy.copy()
        
        T *= cooling
    
    return best_xy, best_val

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0
    rng = random.Random(42)
    np.random.seed(42)
    
    # Try multiple structured seeds
    seeds = [_hexagonal_seed(), _hexagonal_seed2(), _hexagonal_seed3()]
    
    # Add rotated versions of the hexagonal seed
    for base_seed in [_hexagonal_seed()]:
        for rot_offset in [15, 30, 45, 60]:
            inner_rot = [[x, y, a + rot_offset] for (x, y, a) in base_seed[0]]
            seeds.append((inner_rot, base_seed[1], base_seed[2], base_seed[3] + rot_offset))
    
    # Add random seeds
    for _ in range(10):
        seeds.append(_make_seed(rng))
    
    for seed in seeds:
        if time.time() > budget - 15:
            break
        inner, oc, s, rot = seed
        s_tight = _min_outer_side(inner, oc, rot)
        
        # Local refinement
        res = _try_refine(inner, oc, s_tight, rot, rng, iters=100, budget=budget - 10)
        inner_ls, oc_ls, s_ls, rot_ls = res
        if _valid(inner_ls, oc_ls, s_ls, rot_ls):
            if s_ls < best_score:
                best_score = s_ls
                best = (inner_ls, oc_ls, s_ls, rot_ls)
        
        # Scipy optimization from the refined starting point
        if time.time() < budget - 20:
            init_params = np.array([p for (x, y, a) in inner_ls for p in [x, y, a]])
            result = _scipy_optimize_full(init_params, rot_ls, oc_ls, budget - 15, maxiter=300)
            if result is not None:
                pts = result.reshape(N, 3)
                inner_sc = [[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])] for i in range(N)]
                s_sc = _min_outer_side(inner_sc, oc_ls, rot_ls)
                if _valid(inner_sc, oc_ls, s_sc, rot_ls):
                    if s_sc < best_score:
                        best_score = s_sc
                        best = (inner_sc, oc_ls, s_sc, rot_ls)
        
        # Cluster optimization (all same rotation)
        if time.time() < budget - 25:
            init_xy = np.array([[p[0], p[1]] for p in inner_ls])
            cluster_result = _scipy_optimize_cluster(init_xy, rot_ls, oc_ls, budget - 20, maxiter=300)
            if cluster_result is not None:
                inner_cl = [[float(cluster_result[i, 0]), float(cluster_result[i, 1]), rot_ls] for i in range(N)]
                s_cl = _min_outer_side(inner_cl, oc_ls, rot_ls)
                if _valid(inner_cl, oc_ls, s_cl, rot_ls):
                    if s_cl < best_score:
                        best_score = s_cl
                        best = (inner_cl, oc_ls, s_cl, rot_ls)
    
    # Simulated annealing on the best cluster found
    if best is not None and time.time() < budget - 10:
        inner, oc, s, rot = best
        best_xy, best_val = _anneal_cluster(inner, oc, rot, rng, budget - 5)
        inner_anneal = [[float(best_xy[i, 0]), float(best_xy[i, 1]), rot] for i in range(N)]
        s_anneal = _min_outer_side(inner_anneal, oc, rot)
        if _valid(inner_anneal, oc, s_anneal, rot):
            if s_anneal < best_score:
                best_score = s_anneal
                best = (inner_anneal, oc, s_anneal, rot)
    
    # Additional random restarts if time permits
    restarts = 0
    while time.time() < budget - 3.0 and (best is None or best_score > 4.8):
        if time.time() > budget - 3.0:
            break
        restarts += 1
        inner, oc, s, rot = _make_seed(rng)
        s_tight = _min_outer_side(inner, oc, rot)
        res = _try_refine(inner, oc, s_tight, rot, rng, iters=80, budget=budget - 3)
        inner, oc, s, rot = res
        if _valid(inner, oc, s, rot):
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)
        if restarts >= 12:
            break
    
    # Final strict validation
    if best is not None:
        inner, oc, s, rot = best
        s_tight = _min_outer_side(inner, oc, rot)
        if _valid(inner, oc, s_tight, rot):
            return inner, (oc[0], oc[1]), s_tight, rot
        else:
            # Result is invalid, fall back to valid seed
            pass

    # Fallback to a known valid hexagonal lattice packing
    inner, oc, s, rot = _hexagonal_seed()
    s_tight = _min_outer_side(inner, oc, rot)
    return inner, (oc[0], oc[1]), s_tight, rot
