import time
import math

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _spiral_order(limit):
    """Generate lattice points in spiral order (by hex distance, then angle)."""
    R = limit
    cands = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            hd = _hexdist(a,b)
            # Convert to Cartesian for angle sorting
            x = a + 0.5*b
            y = 0.8660254037844386*b
            ang = math.atan2(y, x)
            cands.append((hd, ang, a, b))
    cands.sort()
    out = []
    for hd, ang, a, b in cands:
        out.append((a,b))
    return out

def _verify_and_fix(pts, n=32):
    """Verify the configuration is valid and has 76 edges. If not, try to fix."""
    s = set(pts)
    if len(s) != n:
        return None
    e = _count_edges(pts)
    if e >= 76:
        return pts
    return None

def construct():
    n = 32
    
    # Strategy: Use spiral order to get a compact hexagonal shape
    # The first 32 points in spiral order should form a near-hexagonal shape
    # with maximum edge density (76 edges = Harborth optimum)
    
    order = _spiral_order(10)
    
    # Take first 32 points
    base = order[:n]
    e = _count_edges(base)
    
    if e >= 76:
        return [[a,b] for (a,b) in base]
    
    # If spiral order doesn't give 76, try a few variations
    # Try removing the center and taking next points
    for skip in range(0, 5):
        if skip == 0:
            continue
        variant = order[skip:skip+n]
        e2 = _count_edges(variant)
        if e2 >= 76:
            return [[a,b] for (a,b) in variant]
    
    # Fallback: Use the known optimal hexagonal construction
    # Hexagon of radius 3 has 37 points. Remove 5 lowest-degree boundary points.
    pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if _hexdist(a,b) <= 3:
                pts.append((a,b))
    s = set(pts)
    deg = {}
    for p in pts:
        d = 0
        for da,db in _NEIGH:
            if (p[0]+da, p[1]+db) in s:
                d += 1
        deg[p] = d
    pts_sorted = sorted(pts, key=lambda p: (deg[p], _hexdist(p[0],p[1])))
    to_remove = set(pts_sorted[:5])
    result = [p for p in pts if p not in to_remove]
    e3 = _count_edges(result)
    if e3 >= 76:
        return [[a,b] for (a,b) in result]
    
    # Final fallback: local search from best candidate
    import random
    deadline = time.time() + 110
    best = list(base)
    best_e = e
    in_set = set(best)
    
    pool = set()
    for (a,b) in best:
        for da in range(-4, 5):
            for db in range(-4, 5):
                na, nb = a+da, b+db
                if _hexdist(na, nb) <= 8:
                    pool.add((na, nb))
    pool = list(pool)
    
    rng = random.Random(0)
    while time.time() < deadline:
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            continue
        
        edges_lost = 0
        for da, db in _NEIGH:
            if (cur[0]+da, cur[1]+db) in in_set:
                edges_lost += 1
        edges_gained = 0
        for da, db in _NEIGH:
            nb = (cand[0]+da, cand[1]+db)
            if nb in in_set and nb != cur:
                edges_gained += 1
        
        delta = edges_gained - edges_lost
        if delta > 0:
            best_e += delta
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
            if best_e >= 76:
                break
    
    return [[a,b] for (a,b) in best]
