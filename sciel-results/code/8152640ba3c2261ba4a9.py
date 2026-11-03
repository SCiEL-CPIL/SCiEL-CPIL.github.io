"""Packing 12 unit regular hexagons inside a regular hexagon; minimize outer side length."""
import numpy as np
import time

N = 12
DEADLINE = 140.0  # safety margin before the 150s kill

# ---- Exact geometry of a unit regular hexagon (circumradius = 1) ----
# Vertices at angles 0,60,...,300 (pointy-top).
_VERT_ANGLES = np.deg2rad(np.arange(0, 360, 60))
_UNIT_VERTS = np.stack([np.cos(_VERT_ANGLES), np.sin(_VERT_ANGLES)], axis=1)  # (6,2)

def hex_verts(cx, cy, ang_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) rotated by ang_deg."""
    a = np.deg2rad(ang_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (R @ _UNIT_VERTS.T).T + np.array([cx, cy])

def hex_support(cx, cy, ang_deg, dirs):
    """Support function h(u) = max_v v·u of the hexagon, for a set of unit directions dirs (k,2)."""
    v = hex_verts(cx, cy, ang_deg)
    return v @ dirs.T  # (6,k) -> max over rows

def min_separation(c1, a1, c2, a2):
    """Minimum distance between two unit hexagons (>=0 if overlapping).
    = max over unit u of ( - (h1(u) + h2(-u) - (c2-c1)·u) ) ... 
    Actually distance = max(0, min over u of ( (c2-c1)·u - h1(u) - h2(-u) ) )? 
    For two convex sets A,B: dist(A,B) = max(0, min_{u unit} ( (c2-c1)·u - h_A(u) - h_B(-u) ) ).
    Sample u over 12 directions (enough for regular hexagons)."""
    d = np.array(c2) - np.array(c1)
    # sample directions: 12 evenly spaced
    us = np.deg2rad(np.arange(0, 360, 30))
    U = np.stack([np.cos(us), np.sin(us)], axis=1)
    h1 = hex_support(c1, a1, U).max(axis=0)      # h_A(u)
    h2m = hex_support(c2, a2, -U).max(axis=0)    # h_B(-u)
    vals = d @ U.T - h1 - h2m
    return max(0.0, -vals.min())  # if min vals <0, overlap depth = -min

def overlap_depth(c1, a1, c2, a2):
    """Return how much the two hexagons overlap (0 if disjoint)."""
    d = np.array(c2) - np.array(c1)
    us = np.deg2rad(np.arange(0, 360, 30))
    U = np.stack([np.cos(us), np.sin(us)], axis=1)
    h1 = hex_support(c1, a1, U).max(axis=0)
    h2m = hex_support(c2, a2, -U).max(axis=0)
    vals = d @ U.T - h1 - h2m
    return max(0.0, -vals.min())

# ---- Outer hexagon feasibility ----
# Outer regular hexagon side S, orientation ang_out, center (ox,oy).
# A point p is inside iff for all 6 side-normals n_k, (p - o)·n_k <= inradius_out,
# where inradius_out = S*sqrt(3)/2 and n_k are unit normals to the sides.
def outer_inradius(S):
    return S * np.sqrt(3.0) / 2.0

def outer_normals(ang_out_deg):
    a = np.deg2rad(ang_out_deg)
    base = np.deg2rad(np.arange(0, 360, 60)) + a
    return np.stack([np.cos(base), np.sin(base)], axis=1)

def boundary_violation(p, o, S, nms):
    """Sum of squared amounts by which p violates the 6 half-planes (0 if inside)."""
    d = np.array(p) - np.array(o)
    r = outer_inradius(S)
    vals = d @ nms.T  # (6,)
    over = np.maximum(vals - r, 0.0)
    return float(np.sum(over * over))

def total_penalty(centers, angs, o, S, ang_out):
    nms = outer_normals(ang_out)
    pen = 0.0
    for p, a in zip(centers, angs):
        pen += boundary_violation(p, o, S, nms)
    for i in range(N):
        for j in range(i+1, N):
            od = overlap_depth(centers[i], angs[i], centers[j], angs[j])
            pen += od * od
    return pen

def is_feasible(centers, angs, o, S, ang_out, tol=1e-7):
    nms = outer_normals(ang_out)
    r = outer_inradius(S)
    for p in centers:
        d = np.array(p) - np.array(o)
        if np.any(d @ nms.T > r + tol):
            return False
    for i in range(N):
        for j in range(i+1, N):
            if overlap_depth(centers[i], angs[i], centers[j], angs[j]) > tol:
                return False
    return True

def local_search(centers, angs, o, S, ang_out, iters=1200, lr0=0.15, rng=None):
    rng = rng or np.random.default_rng()
    centers = [list(c) for c in centers]
    angs = list(angs)
    nms = outer_normals(ang_out)
    r = outer_inradius(S)
    best = (centers, angs)
    best_pen = total_penalty(centers, angs, o, S, ang_out)
    cur_pen = best_pen
    for it in range(iters):
        lr = lr0 * (0.995 ** it)
        improved = False
        for i in range(N):
            # try moving center i
            x, y = centers[i]
            a = angs[i]
            # gradient approx for position (use finite diff on penalty w.r.t. this center)
            # boundary grad
            d = np.array([x, y]) - np.array(o)
            gx = 0.0; gy = 0.0
            vals = d @ nms.T
            for k in range(6):
                if vals[k] > r:
                    gx += 2*(vals[k]-r)*nms[k,0]; gy += 2*(vals[k]-r)*nms[k,1]
            # pairwise overlap grad (finite diff)
            eps = 1e-3
            for j in range(N):
                if j == i: continue
                # numeric grad of overlap_depth^2 w.r.t. x,y
                o0 = overlap_depth(centers[i], a, centers[j], angs[j])
                oxx = overlap_depth([x+eps,y], a, centers[j], angs[j])
                oyy = overlap_depth([x,y+eps], a, centers[j], angs[j])
                gx += 2*(oxx*oxx - o0*o0)/eps
                gy += 2*(oyy*oyy - o0*o0)/eps
            nx = x - lr*gx; ny = y - lr*gy
            centers[i] = [nx, ny]
            npen = total_penalty(centers, angs, o, S, ang_out)
            if npen < cur_pen:
                cur_pen = npen
                if npen < best_pen:
                    best_pen = npen
                    best = ([list(c) for c in centers], list(angs))
                improved = True
            else:
                centers[i] = [x, y]
            # occasionally rotate
            if rng.random() < 0.02:
                angs[i] = a + rng.choice([30.0, -30.0])
                npen2 = total_penalty(centers, angs, o, S, ang_out)
                if npen2 < cur_pen:
                    cur_pen = npen2
                    if npen2 < best_pen:
                        best_pen = npen2
                        best = ([list(c) for c in centers], list(angs))
                    improved = True
                else:
                    angs[i] = a
        if not improved and rng.random() < 0.6:
            idx = rng.integers(N)
            centers[idx] = [centers[idx][0]+rng.normal(0,0.3), centers[idx][1]+rng.normal(0,0.3)]
            npen = total_penalty(centers, angs, o, S, ang_out)
            if npen < cur_pen + 1.0:
                cur_pen = npen
    return best[0], best[1], best_pen

def try_fit(S, time_budget, o=(0.0,0.0), ang_out=0.0, rng=None):
    rng = rng or np.random.default_rng()
    t0 = time.time()
    r = outer_inradius(S)
    R_circ = 2.0*r/np.sqrt(3.0)
    best = None; best_pen = np.inf
    while time.time() - t0 < time_budget:
        centers = []; angs = []
        for _ in range(N):
            while True:
                ang = rng.uniform(0, 2*np.pi)
                rad = rng.uniform(0, R_circ)
                x, y = o[0]+rad*np.cos(ang), o[1]+rad*np.sin(ang)
                # check inside outer
                d = np.array([x,y]) - np.array(o)
                nms = outer_normals(ang_out)
                if np.all(d @ nms.T <= r + 1e-9):
                    centers.append([x,y]); angs.append(0.0); break
        pen = total_penalty(centers, angs, o, S, ang_out)
        if pen < best_pen:
            best_pen = pen; best = (centers, angs)
        if pen < 1e-6:
            return centers, angs
        c2, a2, p2 = local_search(centers, angs, o, S, ang_out, iters=1000, lr0=0.15, rng=rng)
        if p2 < best_pen:
            best_pen = p2; best = (c2, a2)
        if p2 < 1e-7:
            return c2, a2
    if best_pen < 1e-7:
        return best[0], best[1]
    return None

def construct():
    t_start = time.time()
    rng = np.random.default_rng(7)
    o = (0.0, 0.0); ang_out = 0.0

    # find a feasible upper bound
    best_S = None; best_c = None; best_a = None
    for S in [12.0, 10.0, 8.0, 7.0, 6.5, 6.0, 5.5]:
        if time.time() - t_start > DEADLINE - 25:
            break
        alloc = min(8.0, DEADLINE - 20 - (time.time()-t_start))
        if alloc <= 0: break
        res = try_fit(S, alloc, o, ang_out, rng)
        if res is not None:
            best_S = S; best_c, best_a = res
            break

    if best_c is None:
        # fallback: grid, large S
        centers = [[float(i), float(j)] for i in range(3) for j in range(4)]
        return [[c[0], c[1], 0.0] for c in centers], (0.0,0.0), 15.0, 0.0

    # binary search down
    lo, hi = 3.0, best_S
    for _ in range(5):
        if time.time() - t_start > DEADLINE - 15:
            break
        mid = 0.5*(lo+hi)
        alloc = min(12.0, DEADLINE - 12 - (time.time()-t_start))
        if alloc <= 0: break
        res = try_fit(mid, alloc, o, ang_out, rng)
        if res is not None:
            hi = mid; best_S = mid; best_c, best_a = res
        else:
            lo = mid

    # final polish
    if time.time() - t_start < DEADLINE - 3:
        alloc = DEADLINE - 3 - (time.time()-t_start)
        res = try_fit(best_S, alloc, o, ang_out, rng)
        if res is not None:
            best_c, best_a = res

    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(best_c, best_a)]
    return inner, (0.0, 0.0), float(best_S), 0.0
