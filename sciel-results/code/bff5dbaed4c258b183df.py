import math
import time
import random
import numpy as np

N = 11
EPS = 1e-9
START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _outer_vertices(cx, cy, s, rot):
    pts = []
    for k in range(6):
        a = math.radians(rot) + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    pts = []
    a0 = math.radians(ang_deg)
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

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
    # SAT: two convex polygons are disjoint iff there is a separating axis
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            nx /= norm
            ny /= norm
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            if maxA + 1e-9 < minB or maxB + 1e-9 < minA:
                return False  # separated
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    outer_pts = _outer_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _min_outer_side(inner, outer_c, rot):
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    max_proj = 0.0
    for (cx, cy, ang) in inner:
        verts = _inner_hex(cx, cy, ang)
        for (vx, vy) in verts:
            dx = vx - outer_c[0]
            dy = vy - outer_c[1]
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    
    S = 2.0 * max_proj / math.sqrt(3.0)
    return S

def _penalty_objective(inner, outer_c, s, rot):
    """
    Continuous penalty: sum of penetration depths for overlaps + boundary violations.
    Returns 0.0 if valid.
    """
    outer_pts = _outer_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    
    penalty = 0.0
    
    # Boundary violations: for each inner vertex, if outside outer, add distance
    for poly in polys:
        for pt in poly:
            if not _point_in_poly(pt, outer_pts):
                # Find closest point on outer boundary
                min_dist = float('inf')
                for i in range(len(outer_pts)):
                    p1 = outer_pts[i]
                    p2 = outer_pts[(i+1) % len(outer_pts)]
                    # Project pt onto segment p1-p2
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    len2 = dx*dx + dy*dy
                    if len2 < 1e-12:
                        continue
                    t = ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / len2
                    t = max(0.0, min(1.0, t))
                    qx = p1[0] + t*dx
                    qy = p1[1] + t*dy
                    dist = math.sqrt((pt[0]-qx)**2 + (pt[1]-qy)**2)
                    if dist < min_dist:
                        min_dist = dist
                penalty += min_dist * 10.0
    
    # Pairwise overlap: use penetration depth approximation via SAT
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            A = polys[i]
            B = polys[j]
            # Check if overlapping via SAT, and estimate penetration
            min_pen = float('inf')
            overlapping = True
            for poly in (A, B):
                n = len(poly)
                for k in range(n):
                    ax = poly[(k+1) % n][0] - poly[k][0]
                    ay = poly[(k+1) % n][1] - poly[k][1]
                    nx, ny = -ay, ax
                    norm = math.sqrt(nx*nx + ny*ny)
                    if norm < 1e-12:
                        continue
                    nx /= norm
                    ny /= norm
                    minA = min(p[0]*nx + p[1]*ny for p in A)
                    maxA = max(p[0]*nx + p[1]*ny for p in A)
                    minB = min(p[0]*nx + p[1]*ny for p in B)
                    maxB = max(p[0]*nx + p[1]*ny for p in B)
                    if maxA + 1e-9 < minB or maxB + 1e-9 < minA:
                        overlapping = False
                        break
                    else:
                        pen = maxA - minB if maxA > minB else maxB - minA
                        if pen < min_pen:
                            min_pen = pen
                if not overlapping:
                    break
            if overlapping:
                penalty += min_pen * 10.0
    
    return penalty

def _lattice_seed(topo_indices, outer_rot=0.0):
    A = (math.sqrt(3), 0.0)
    B = (math.sqrt(3)/2, 1.5)
    
    centers = []
    for i, j in topo_indices:
        cx = i * A[0] + j * B[0]
        cy = i * A[1] + j * B[1]
        centers.append((cx, cy))
    
    rot_rad = math.radians(outer_rot)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    rotated_centers = []
    for (x, y) in centers:
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        rotated_centers.append((rx, ry))
        
    inner = [[c[0], c[1], 0.0] for c in rotated_centers]
    s = _min_outer_side(inner, (0.0, 0.0), 0.0)
    return inner, (0.0, 0.0), s, 0.0

def _anneal(inner, outer_c, s, rot, rng, budget):
    """
    Simulated annealing on continuous penalty objective.
    Variables: inner positions/angles, outer center, side, rotation.
    """
    # Flatten to vector
    def to_vec(inner, oc, s, rot):
        v = []
        for (x, y, a) in inner:
            v.extend([x, y, a])
        v.extend([oc[0], oc[1], s, rot])
        return v
    
    def from_vec(v):
        inner = []
        for i in range(N):
            inner.append([v[3*i], v[3*i+1], v[3*i+2]])
        oc = (v[3*N], v[3*N+1])
        s = v[3*N+2]
        rot = v[3*N+3]
        return inner, oc, s, rot
    
    cur_v = to_vec(inner, outer_c, s, rot)
    cur_pen = _penalty_objective(*from_vec(cur_v))
    
    best_v = cur_v[:]
    best_pen = cur_pen
    
    T = 1.0
    T_min = 1e-6
    cool_rate = 0.9995
    step = 0.15
    
    while time.time() < budget:
        # Perturb
        new_v = cur_v[:]
        for i in range(len(new_v)):
            if i < 3*N:  # positions and angles
                new_v[i] += rng.gauss(0, step)
            elif i == 3*N or i == 3*N+1:  # outer center
                new_v[i] += rng.gauss(0, step*0.5)
            elif i == 3*N+2:  # side
                new_v[i] += rng.gauss(0, step*0.3)
            else:  # rotation
                new_v[i] += rng.gauss(0, step*0.2)
        
        new_pen = _penalty_objective(*from_vec(new_v))
        
        if new_pen < cur_pen or rng.random() < math.exp(-(new_pen - cur_pen) / max(T, T_min)):
            cur_v = new_v
            cur_pen = new_pen
            if cur_pen < best_pen:
                best_v = cur_v[:]
                best_pen = cur_pen
        
        T = max(T * cool_rate, T_min)
        step *= 0.9999
    
    return from_vec(best_v)

def _gradient_descent(inner, outer_c, s, rot, rng, budget):
    """
    Simple gradient-free local search using Nelder-Mead-like simplex on penalty.
    """
    # Use scipy minimize if available
    try:
        from scipy.optimize import minimize
    except ImportError:
        return inner, outer_c, s, rot
    
    def to_vec(inner, oc, s, rot):
        v = []
        for (x, y, a) in inner:
            v.extend([x, y, a])
        v.extend([oc[0], oc[1], s, rot])
        return np.array(v, dtype=float)
    
    def from_vec(v):
        inner = []
        for i in range(N):
            inner.append([v[3*i], v[3*i+1], v[3*i+2]])
        oc = (v[3*N], v[3*N+1])
        s = v[3*N+2]
        rot = v[3*N+3]
        return inner, oc, s, rot
    
    def objective(v):
        inner, oc, s, rot = from_vec(v)
        if s < 1.0:
            return 1e6
        return _penalty_objective(inner, oc, s, rot)
    
    x0 = to_vec(inner, outer_c, s, rot)
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if result.fun < _penalty_objective(*from_vec(x0)):
        return from_vec(result.x)
    return inner, outer_c, s, rot

def _make_seed(rng):
    inner = [[rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0, 60)] for _ in range(N)]
    return inner, (0.0, 0.0), 12.0, 0.0

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0  # safety margin
    rng = random.Random(42)

    # Structured lattice seeds
    # Topology: central 7 + 4 more
    t1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,0), (0,-2)]
    t2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    t3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,2), (-1,1)]
    t4 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (-1,1), (-2,0)]
    t5 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (2,-1), (-2,1)]
    t6 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (0,2), (-1,1), (-1,-1)]
    
    topologies = [t1, t2, t3, t4, t5, t6]
    
    seeds = []
    for topo in topologies:
        for rot in [0, 30, 60]:
            seeds.append(_lattice_seed(topo, rot))
    
    # Add random seeds
    for _ in range(8):
        seeds.append(_make_seed(rng))

    # Phase 1: Annealing on each seed
    for seed in seeds:
        if time.time() > budget:
            break
        inner, oc, s, rot = seed
        sub_budget = min(budget, time.time() + 10.0)
        res = _anneal(inner, oc, s, rot, rng, sub_budget)
        inner, oc, s, rot = res
        pen = _penalty_objective(inner, oc, s, rot)
        if pen < 1e-6:
            # Valid, compute true min outer side
            opt_s = _min_outer_side(inner, oc, rot)
            if _valid(inner, oc, opt_s, rot):
                s = opt_s
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)

    # Phase 2: Nelder-Mead refinement on best found
    if best is not None and time.time() < budget - 5.0:
        inner, oc, s, rot = best
        sub_budget = min(budget, time.time() + 30.0)
        res = _gradient_descent(inner, oc, s, rot, rng, sub_budget)
        inner, oc, s, rot = res
        pen = _penalty_objective(inner, oc, s, rot)
        if pen < 1e-6:
            opt_s = _min_outer_side(inner, oc, rot)
            if _valid(inner, oc, opt_s, rot):
                s = opt_s
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)

    # Phase 3: More random restarts if time remains
    restarts = 0
    while time.time() < budget - 10.0:
        restarts += 1
        inner, oc, s, rot = _make_seed(rng)
        sub_budget = min(budget - 5.0, time.time() + 8.0)
        res = _anneal(inner, oc, s, rot, rng, sub_budget)
        inner, oc, s, rot = res
        pen = _penalty_objective(inner, oc, s, rot)
        if pen < 1e-6:
            opt_s = _min_outer_side(inner, oc, rot)
            if _valid(inner, oc, opt_s, rot):
                s = opt_s
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)
        if restarts >= 50:
            break

    # Phase 4: Final Nelder-Mead on best
    if best is not None and time.time() < budget - 3.0:
        inner, oc, s, rot = best
        sub_budget = min(budget, time.time() + 20.0)
        res = _gradient_descent(inner, oc, s, rot, rng, sub_budget)
        inner, oc, s, rot = res
        pen = _penalty_objective(inner, oc, s, rot)
        if pen < 1e-6:
            opt_s = _min_outer_side(inner, oc, rot)
            if _valid(inner, oc, opt_s, rot):
                s = opt_s
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)

    if best is None:
        # Fallback: large valid configuration
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    inner, oc, s, rot = best
    return inner, (oc[0], oc[1]), s, rot
