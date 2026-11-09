"""
Combined approach for maximizing unit-distance edges on a triangular lattice.
Uses multiple initialization strategies and fast incremental local search.
"""
import time
import math
import random

# EVOLVE-BLOCK-START

# Neighbors in axial coordinates (a, b) for triangular lattice
# omega = exp(i*pi/3)
# Neighbors of (a,b) are:
# (a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)
_NEIGH = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]

def _hexdist(a, b):
    """Hexagonal distance in axial coordinates."""
    return (abs(a) + abs(b) + abs(a + b)) // 2

def _count_edges(pts):
    """Count the number of pairs at distance exactly 1."""
    s = set(pts)
    cnt = 0
    for (a, b) in s:
        for da, db in _NEIGH:
            if (a + da, b + db) in s:
                cnt += 1
    return cnt // 2

def _generate_hexagon(radius):
    """Generate all points in a hexagon of given radius."""
    pts = []
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            if _hexdist(a, b) <= radius:
                pts.append((a, b))
    return pts

def _known_optimal_init():
    """
    Initialize by taking a hexagon of radius 3 (37 points) and removing
    5 points with the lowest degree to minimize edge loss.
    """
    pts = _generate_hexagon(3)
    s = set(pts)
    
    # Calculate degree of each point
    deg = {}
    for p in pts:
        d = 0
        for da, db in _NEIGH:
            if (p[0] + da, p[1] + db) in s:
                d += 1
        deg[p] = d
        
    # Sort by degree (ascending), then by hex distance (descending) to remove outer low-degree points
    # We want to remove points that contribute the least to the edge count.
    # Removing a point with degree d loses d edges.
    # So we remove the 5 points with the smallest degree.
    # If ties, prefer removing points further from center? Or closer?
    # Usually, corners have degree 2, edges have degree 3, inner have 4/6.
    # Removing corners (deg 2) is best.
    pts_sorted = sorted(pts, key=lambda p: (deg[p], _hexdist(p[0], p[1])))
    
    to_remove = set(pts_sorted[:5])
    result = [p for p in pts if p not in to_remove]
    return result

def _spiral_init(n=32, limit=12):
    """
    Initialize by taking the first n points in a spiral order.
    """
    cands = []
    for a in range(-limit, limit + 1):
        for b in range(-limit, limit + 1):
            hd = _hexdist(a, b)
            if hd > limit:
                continue
            x = a + 0.5 * b
            y = 0.8660254037844386 * b
            ang = math.atan2(y, x)
            cands.append((hd, ang, a, b))
    cands.sort()
    out = [(a, b) for hd, ang, a, b in cands[:n]]
    return out

def _greedy_init(n=32, limit=6):
    """
    Greedily add points that maximize the number of new edges.
    """
    rng = random.Random(42)
    in_set = set()
    pts = []
    pool = []
    for a in range(-limit, limit + 1):
        for b in range(-limit, limit + 1):
            if _hexdist(a, b) <= limit:
                pool.append((a, b))
    rng.shuffle(pool)
    
    for _ in range(n):
        best_pt = None
        best_gain = -1
        for cand in pool:
            if cand in in_set:
                continue
            gain = 0
            for da, db in _NEIGH:
                if (cand[0] + da, cand[1] + db) in in_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_pt = cand
            elif gain == best_gain and best_pt is not None:
                # Tie-break: prefer closer to origin
                if _hexdist(cand[0], cand[1]) < _hexdist(best_pt[0], best_pt[1]):
                    best_pt = cand
        if best_pt is None:
            break
        in_set.add(best_pt)
        pts.append(best_pt)
    return pts

def _local_search(pts, deadline, rng, n=32):
    """
    Perform local search with incremental edge counting.
    Swap points if it increases the edge count.
    """
    best = list(pts)
    in_set = set(best)
    best_e = _count_edges(best)
    
    if best_e >= 76:
        return best, best_e

    # Build a candidate pool for swaps
    # Points should be reasonably close to the current shape
    pool_set = set()
    for (a, b) in best:
        for da in range(-5, 6):
            for db in range(-5, 6):
                na, nb = a + da, b + db
                if _hexdist(na, nb) <= 10:
                    pool_set.add((na, nb))
    pool = list(pool_set)
    
    it = 0
    no_improve = 0
    max_no_improve = 20000 # Stop if stuck for a while
    
    while time.time() < deadline and no_improve < max_no_improve:
        it += 1
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            no_improve += 1
            continue
        
        # Calculate delta in edge count
        # Edges lost: neighbors of cur that are in in_set
        edges_lost = 0
        for da, db in _NEIGH:
            if (cur[0] + da, cur[1] + db) in in_set:
                edges_lost += 1
                
        # Edges gained: neighbors of cand that are in in_set (excluding cur)
        edges_gained = 0
        for da, db in _NEIGH:
            nb = (cand[0] + da, cand[1] + db)
            if nb in in_set and nb != cur:
                edges_gained += 1

        delta = edges_gained - edges_lost
        if delta > 0:
            best_e += delta
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
            no_improve = 0
        else:
            no_improve += 1
            
    return best, best_e

def construct():
    """
    Returns 32 distinct integer pairs [a, b] on the triangular lattice
    (point a + b*omega, omega = exp(i*pi/3)) with |a|,|b| <= 64,
    mutual distance >= 1, maximizing pairs at distance exactly 1.
    Optimum is 76 (Harborth's theorem).
    """
    t0 = time.time()
    deadline = t0 + 120.0  # Safety margin before 150s kill
    
    n = 32
    best_points = None
    best_edges = -1
    
    # Generate initial candidates
    candidates = []
    
    # 1. Known optimal heuristic
    try:
        c1 = _known_optimal_init()
        if len(c1) == n:
            candidates.append(c1)
    except Exception:
        pass
        
    # 2. Spiral init
    try:
        c2 = _spiral_init(n)
        if len(c2) == n:
            candidates.append(c2)
    except Exception:
        pass
        
    # 3. Greedy init
    try:
        c3 = _greedy_init(n)
        if len(c3) == n:
            candidates.append(c3)
    except Exception:
        pass

    # If no candidates generated, use a simple fallback
    if not candidates:
        candidates.append([(i, 0) for i in range(n)])

    # Run local search on each candidate
    rng = random.Random(0)
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Allocate remaining time to each candidate, but prioritize first few
        # Simple strategy: give each candidate 1/3 of the time, or just run until deadline
        # Since local search is fast, we can just run it.
        
        # Reset RNG for reproducibility per candidate if needed, or continue
        # Let's use a fresh RNG for each to diversify search
        local_rng = random.Random(1000 + i)
        
        pts, edges = _local_search(cand, deadline, local_rng, n)
        
        if edges > best_edges:
            best_edges = edges
            best_points = pts
            
        if best_edges == 76:
            break # Found optimum

    # Fallback if something went wrong
    if best_points is None:
        best_points = [(i, 0) for i in range(n)]
        
    # Ensure distinct and valid (should be by construction, but check)
    # The local search maintains validity (distinct points)
    
    return [[a, b] for a, b in best_points]

# EVOLVE-BLOCK-END
