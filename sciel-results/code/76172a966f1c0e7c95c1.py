import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
EPS = 1e-9
START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, s, rot):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation rot (degrees)."""
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    """Unit regular hexagon (side 1) with center (cx,cy) and rotation ang_deg."""
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    """
    Compute the minimum side length S of a regular hexagon centered at outer_c
    with rotation rot that contains all inner hexagons.
    """
    theta_rad = math.radians(rot)
    # Normals to the sides of the outer hexagon
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
    
    # S * sqrt(3)/2 >= max_proj  =>  S >= 2 * max_proj / sqrt(3)
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
    """SAT: return True if convex polygons A and B overlap."""
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
    """
    Create a hexagonal lattice arrangement of 11 unit hexagons.
    For 11 hexagons: 1 center + 6 in first ring + 4 in second ring.
    """
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    # First ring: 6 hexagons at distance sqrt(3) in 6 directions
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    # Second ring: add 4 hexagons at distance 2*sqrt(3) in 4 of the 6 directions
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    
    return inner, (0.0, 0.0), 0.0, 0.0

def _make_seed(rng):
    inner = [[rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 60)] for _ in range(N)]
    return inner, (0.0, 0.0), 10.0, 0.0

def _separation_violation(params):
    """
    Compute the minimum separation violation between all pairs of hexagons.
    Positive value means overlap (bad), negative means separated (good).
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    
    max_viol = 0.0
    for i in range(N):
        for j in range(i+1, N):
            # Compute minimum distance between two convex polygons
            # Using SAT to find the minimum penetration depth
            min_dist = float('inf')
            for poly in (polys[i], polys[j]):
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
                    minA = min(p[0]*nx + p[1]*ny for p in polys[i])
                    maxA = max(p[0]*nx + p[1]*ny for p in polys[i])
                    minB = min(p[0]*nx + p[1]*ny for p in polys[j])
                    maxB = max(p[0]*nx + p[1]*ny for p in polys[j])
                    gap = min(maxA - maxB, maxB - maxA)
                    if gap < min_dist:
                        min_dist = gap
            # If min_dist < 0, they overlap
            if min_dist < 0:
                max_viol = max(max_viol, -min_dist)
    
    return max_viol

def _objective(params, outer_c, rot, target_s):
    """
    Objective function: minimize the outer side length needed,
    with a penalty for overlap violations.
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    # Compute the required outer side
    s_req = _min_outer_side(inner, outer_c, rot)
    
    # Check if within target
    if s_req > target_s:
        penalty_s = (s_req - target_s) ** 2 * 100.0
    else:
        penalty_s = 0.0
    
    # Check overlap violations
    viol = _separation_violation(params)
    penalty_v = viol ** 2 * 1000.0 if viol > 0 else 0.0
    
    # Check if inside outer hexagon
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], target_s, rot)
    penalty_outer = 0.0
    for (x, y, a) in inner:
        pts = _inner_hex(x, y, a)
        for p in pts:
            if not _point_in_poly(p, outer_pts):
                # Compute distance outside
                # Simplified: just add penalty
                penalty_outer += 1.0
    
    return s_req + penalty_s + penalty_v + penalty_outer * 10.0

def _nelder_mead_optimize(inner_init, outer_c, rot, target_s, rng, budget):
    """
    Use Nelder-Mead to optimize the positions and angles of inner hexagons.
    """
    x0 = []
    for (x, y, a) in inner_init:
        x0.extend([x, y, a])
    
    def obj(params):
        return _objective(params, outer_c, rot, target_s)
    
    result = minimize(obj, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
    
    if time.time() > budget:
        return None
    
    best_params = result.x
    best_obj = result.fun
    
    # Extract inner
    inner_opt = []
    for i in range(N):
        inner_opt.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    # Compute actual tight side
    s_tight = _min_outer_side(inner_opt, outer_c, rot)
    
    return inner_opt, outer_c, s_tight, rot

def _binary_search_feasible(inner, outer_c, rot, rng, budget):
    """
    Binary search on the outer side length to find the minimum feasible size.
    Uses Nelder-Mead to check feasibility at each size.
    """
    # Initial bounds
    s_low = 3.0
    s_high = 8.0
    
    # First, find a valid configuration at s_high
    best_inner = None
    for attempt in range(5):
        if time.time() > budget:
            break
        inner_try = _nelder_mead_optimize(inner, outer_c, rot, s_high, rng, budget)
        if inner_try is not None:
            inner_opt, oc, s, r = inner_try
            if _valid(inner_opt, oc, s, r) and s <= s_high + 0.1:
                best_inner = inner_opt
                break
    
    if best_inner is None:
        return None
    
    # Binary search
    for _ in range(30):
        if time.time() > budget - 5.0:
            break
        s_mid = (s_low + s_high) / 2.0
        
        inner_try = _nelder_mead_optimize(best_inner, outer_c, rot, s_mid, rng, budget)
        if inner_try is not None:
            inner_opt, oc, s, r = inner_try
            if _valid(inner_opt, oc, s, r) and s <= s_mid + 0.05:
                s_high = s_mid
                best_inner = inner_opt
            else:
                s_low = s_mid
        else:
            s_low = s_mid
    
    # Final result
    if best_inner is not None:
        s_tight = _min_outer_side(best_inner, outer_c, rot)
        if _valid(best_inner, outer_c, s_tight, rot):
            return best_inner, outer_c, s_tight, rot
    
    return None

def _optimize_cluster(inner_init, outer_c, rot, rng, budget):
    """
    Main optimization: use Nelder-Mead to optimize the cluster,
    then binary search on the outer side length.
    """
    # First, get a good initial configuration using Nelder-Mead with a loose bound
    s_loose = 6.0
    result = _nelder_mead_optimize(inner_init, outer_c, rot, s_loose, rng, budget)
    
    if result is None:
        return None
    
    inner_opt, oc, s, r = result
    
    # Now binary search for the minimum feasible size
    bs_result = _binary_search_feasible(inner_opt, oc, r, rng, budget)
    
    if bs_result is not None:
        return bs_result
    
    # Fallback: use the result from Nelder-Mead
    s_tight = _min_outer_side(inner_opt, oc, r)
    if _valid(inner_opt, oc, s_tight, r):
        return inner_opt, oc, s_tight, r
    
    return None

def _try_refine(inner, outer_c, s, rot, rng, iters, budget):
    best = (inner, outer_c, s, rot)
    best_pen = 1.0 if not _valid(inner, outer_c, s, rot) else 0.0
    cur = (inner, outer_c, s, rot)
    cur_pen = best_pen
    
    # Tighten s first
    s_tight = _min_outer_side(cur[0], cur[1], cur[3])
    if s_tight < cur[2]:
        cur = (cur[0], cur[1], s_tight, cur[3])
        if _valid(cur[0], cur[1], cur[2], cur[3]):
            best = cur
            best_pen = 0.0
            cur_pen = 0.0
    
    it = 0
    while it < iters:
        if time.time() > budget:
            break
        it += 1
        # Perturb
        cand = [[x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), a + rng.uniform(-15, 15)] for (x, y, a) in cur[0]]
        cand_oc = (cur[1][0] + rng.uniform(-0.2, 0.2), cur[1][1] + rng.uniform(-0.2, 0.2))
        cand_rot = cur[3] + rng.uniform(-10, 10)
        
        # Compute tight s for this candidate
        cand_s = _min_outer_side(cand, cand_oc, cand_rot)
        
        if _valid(cand, cand_oc, cand_s, cand_rot):
            cur = (cand, cand_oc, cand_s, cand_rot)
            cur_pen = 0.0
            if cand_s < best[2]:
                best = cur
                best_pen = 0.0
        else:
            pass
    
    return best

def _anneal_cluster(inner, outer_c, rot, rng, budget):
    """
    Simulated annealing on the global rotation of the cluster and the outer center.
    The cluster internal structure is fixed from the seed.
    We minimize the outer side length by rotating the whole cluster and shifting the outer center.
    """
    # Fix the cluster: all inner hexagons move together with the same rotation offset
    # Actually, let's parameterize: each inner hexagon has a base position, and we apply a global rotation
    # and a global translation to the entire cluster, then find the optimal outer hexagon.
    
    # Base positions (relative to cluster center)
    base_positions = [(x, y, a) for (x, y, a) in inner]
    cluster_cx = sum(x for x, y, a in base_positions) / N
    cluster_cy = sum(y for x, y, a in base_positions) / N
    
    # Normalize to center at origin
    base_rel = [(x - cluster_cx, y - cluster_cy, a) for (x, y, a) in base_positions]
    
    best_s = float('inf')
    best_state = None
    
    # Simulated annealing parameters
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    
    cur_global_rot = 0.0  # radians
    cur_offset_x = 0.0
    cur_offset_y = 0.0
    cur_s = _min_outer_side(inner, outer_c, rot)
    
    # Initial evaluation
    if _valid(inner, outer_c, cur_s, rot):
        best_s = cur_s
        best_state = (inner, outer_c, cur_s, rot)
    
    it = 0
    while T > T_min:
        if time.time() > budget:
            break
        it += 1
        
        # Perturb global rotation and offset
        d_rot = rng.gauss(0, 0.1)
        d_ox = rng.gauss(0, 0.2)
        d_oy = rng.gauss(0, 0.2)
        
        new_global_rot = cur_global_rot + d_rot
        new_offset_x = cur_offset_x + d_ox
        new_offset_y = cur_offset_y + d_oy
        
        # Apply global rotation to all inner hexagons
        cos_r = math.cos(new_global_rot)
        sin_r = math.sin(new_global_rot)
        
        new_inner = []
        for (bx, by, ba) in base_rel:
            # Rotate position
            nx = bx * cos_r - by * sin_r + new_offset_x + cluster_cx
            ny = bx * sin_r + by * cos_r + new_offset_y + cluster_cy
            # Rotate orientation
            na = ba + math.degrees(new_global_rot)
            new_inner.append([nx, ny, na])
        
        # Compute tight outer side
        new_s = _min_outer_side(new_inner, outer_c, rot)
        
        # Check validity
        if _valid(new_inner, outer_c, new_s, rot):
            # Accept if better
            if new_s < cur_s:
                cur_global_rot = new_global_rot
                cur_offset_x = new_offset_x
                cur_offset_y = new_offset_y
                cur_s = new_s
                if new_s < best_s:
                    best_s = new_s
                    best_state = (new_inner, outer_c, new_s, rot)
            else:
                # Accept with probability exp(-(new_s - cur_s) / T)
                delta = new_s - cur_s
                if delta < 0 or random.random() < math.exp(-delta / T):
                    cur_global_rot = new_global_rot
                    cur_offset_x = new_offset_x
                    cur_offset_y = new_offset_y
                    cur_s = new_s
        else:
            # Invalid configuration: accept with lower probability
            delta = 1.0  # penalty
            if random.random() < math.exp(-delta / T):
                cur_global_rot = new_global_rot
                cur_offset_x = new_offset_x
                cur_offset_y = new_offset_y
                cur_s = new_s
        
        T *= cooling
    
    if best_state is not None:
        return best_state
    else:
        return (inner, outer_c, cur_s, rot)

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0
    rng = random.Random(42)
    
    # Start with hexagonal seed
    inner_hex, oc_hex, s_hex, rot_hex = _hexagonal_seed()
    s_hex_tight = _min_outer_side(inner_hex, oc_hex, rot_hex)
    
    if _valid(inner_hex, oc_hex, s_hex_tight, rot_hex):
        best_score = s_hex_tight
        best = (inner_hex, oc_hex, s_hex_tight, rot_hex)
    
    # Try Nelder-Mead optimization from the hexagonal seed
    if time.time() < budget - 20.0:
        nm_result = _optimize_cluster(inner_hex, oc_hex, rot_hex, rng, budget - 10.0)
        if nm_result is not None:
            inner_nm, oc_nm, s_nm, rot_nm = nm_result
            if _valid(inner_nm, oc_nm, s_nm, rot_nm) and s_nm < best_score:
                best_score = s_nm
                best = (inner_nm, oc_nm, s_nm, rot_nm)
    
    # Try random seeds with Nelder-Mead
    for i in range(5):
        if time.time() > budget - 15.0:
            break
        inner_rand, oc_rand, s_rand, rot_rand = _make_seed(rng)
        nm_result = _optimize_cluster(inner_rand, oc_rand, rot_rand, rng, budget - 10.0)
        if nm_result is not None:
            inner_nm, oc_nm, s_nm, rot_nm = nm_result
            if _valid(inner_nm, oc_nm, s_nm, rot_nm) and s_nm < best_score:
                best_score = s_nm
                best = (inner_nm, oc_nm, s_nm, rot_nm)
    
    # If we still have time, do more refinement with the best found
    if best is not None and time.time() < budget - 10.0:
        inner_b, oc_b, s_b, rot_b = best
        # Try to further optimize with Nelder-Mead from the current best
        nm_result = _optimize_cluster(inner_b, oc_b, rot_b, rng, budget - 5.0)
        if nm_result is not None:
            inner_nm, oc_nm, s_nm, rot_nm = nm_result
            if _valid(inner_nm, oc_nm, s_nm, rot_nm) and s_nm < best_score:
                best_score = s_nm
                best = (inner_nm, oc_nm, s_nm, rot_nm)
    
    # Final fallback: use the hexagonal seed if nothing better was found
    if best is None:
        s_tight = _min_outer_side(inner_hex, oc_hex, rot_hex)
        return inner_hex, (oc_hex[0], oc_hex[1]), s_tight, rot_hex
    
    inner, oc, s, rot = best
    # Final tightening
    s_tight = _min_outer_side(inner, oc, rot)
    if s_tight < s and _valid(inner, oc, s_tight, rot):
        s = s_tight
    
    return inner, (oc[0], oc[1]), s, rot
