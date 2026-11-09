import time
import math
import random

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def _deg(pt, s):
    a, b = pt
    return sum(1 for da,db in _NEIGH if (a+da, b+db) in s)

def _swap_delta(cur, cand, s):
    """Return change in edge count when replacing cur with cand in set s (cur in s, cand not in s)."""
    lost = _deg(cur, s)
    a, b = cand
    gained = 0
    for da, db in _NEIGH:
        nb = (a+da, b+db)
        if nb in s and nb != cur:
            gained += 1
    return gained - lost

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _spiral_order(limit):
    """Generate lattice points in spiral order (by hex distance, then angle)."""
    R = limit
    cands = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            hd = _hexdist(a,b)
            x = a + 0.5*b
            y = 0.8660254037844386*b
            ang = math.atan2(y, x)
            cands.append((hd, ang, a, b))
    cands.sort()
    out = []
    for hd, ang, a, b in cands:
        out.append((a,b))
    return out

def _build_hexagonal_patch(n):
    """
    Build a compact hexagonal-like shape with n points.
    Prefer complete rings when they fit; only take a partial ring if n
    is not exactly a hexagonal number.  This keeps the shape as close
    to a regular hexagon as possible, minimizing boundary irregularities.
    """
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # H(r) = 1 + 3*r*(r+1)
    def hex_num(r):
        return 1 + 3 * r * (r + 1)

    # Find largest complete hexagon that fits
    r = 0
    while hex_num(r + 1) <= n:
        r += 1

    points = [(0, 0)]
    for ring in range(1, r + 1):
        ring_points = []
        for a in range(-ring, ring + 1):
            for b in range(-ring, ring + 1):
                if _hexdist(a, b) == ring:
                    ring_points.append((a, b))

        rp_sorted = []
        for (a, b) in ring_points:
            x = a + 0.5 * b
            y = 0.8660254037844386 * b
            ang = math.atan2(y, x)
            rp_sorted.append((ang, a, b))
        rp_sorted.sort()

        for ang, a, b in rp_sorted:
            points.append((a, b))

    # If we still need more points, add a partial ring
    if len(points) < n:
        ring = r + 1
        ring_points = []
        for a in range(-ring, ring + 1):
            for b in range(-ring, ring + 1):
                if _hexdist(a, b) == ring:
                    ring_points.append((a, b))

        rp_sorted = []
        for (a, b) in ring_points:
            x = a + 0.5 * b
            y = 0.8660254037844386 * b
            ang = math.atan2(y, x)
            rp_sorted.append((ang, a, b))
        rp_sorted.sort()

        for ang, a, b in rp_sorted:
            if len(points) >= n:
                break
            points.append((a, b))

    return points[:n]

def _greedy_construct(n_points=32, seed=42, max_coord=15):
    """
    Greedy construction: add points one at a time, maximizing new edges.
    Ties are broken by preferring points closer to the centroid.
    """
    rng = random.Random(seed)
    
    # Generate candidate points in a bounding box
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            candidates.append((a, b))
    
    # Start with a random point near origin to break symmetry
    # Actually, starting at (0,0) is usually best for compactness, 
    # but let's try a few offsets or just (0,0).
    # Let's stick to (0,0) for determinism in the greedy step, 
    # but the seed can shuffle the candidate order if we wanted.
    # For now, we will just use the deterministic greedy rule.
    
    placed = [(0, 0)]
    placed_set = {(0, 0)}
    
    for _ in range(n_points - 1):
        best_point = None
        best_new_edges = -1
        best_dist_to_center = float('inf')
        
        # Compute centroid
        cx = sum(p[0] for p in placed) / len(placed)
        cy = sum(p[1] for p in placed) / len(placed)
        
        # To make it faster, we can limit candidates to those near the current set
        # But n=32 is small, so iterating over ~900 candidates is fine.
        
        for pt in candidates:
            if pt in placed_set:
                continue
            
            # Check minimum distance constraint: dist >= 1 from all placed points
            # Since we are on a lattice, distinct points are at least distance 1 apart
            # unless they are the same point. So this check is redundant if we ensure distinctness.
            # Wait, the problem says "mutual distance at least one".
            # On a triangular lattice, the minimum distance between distinct integer pairs (a,b) is 1.
            # So any distinct set of lattice points satisfies the constraint.
            
            # Count new edges formed with already-placed points
            new_edges = 0
            a, b = pt
            for da, db in _NEIGH:
                if (a+da, b+db) in placed_set:
                    new_edges += 1
            
            # Score: primarily new edges, secondary: prefer points closer to cluster center
            # Approximate distance from centroid
            dx = pt[0] - cx
            dy = pt[1] - cy
            # Use lattice distance approximation (hex distance from centroid)
            dist_to_center = _hexdist(int(round(dx)), int(round(dy)))
            # More accurate hex distance from centroid (which is not necessarily integer)
            # But _hexdist is for integers. Let's use Euclidean squared for simplicity in tie-breaking
            dist_sq = dx*dx + dy*dy + dx*dy # Lattice distance squared approximation
            
            if new_edges > best_new_edges or (new_edges == best_new_edges and dist_sq < best_dist_to_center):
                best_new_edges = new_edges
                best_dist_to_center = dist_sq
                best_point = pt
        
        if best_point is None:
            # No valid point found (should not happen with large enough box)
            break
        
        placed.append(best_point)
        placed_set.add(best_point)
    
    return placed

def _try_improve(pts, max_time=10.0, seed=42):
    """Local search to improve edge count via single-point swaps."""
    rng = random.Random(seed)
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Build a pool of candidate points
    # Include points near the current set
    pool = set()
    for (a,b) in best:
        for r in range(0, 10): # Slightly larger radius
            for a2 in range(a-r, a+r+1):
                for b2 in range(b-r, b+r+1):
                    if _hexdist(a2,b2) <= 20:
                        pool.add((a2,b2))
    pool = list(pool)
    
    start = time.time()
    it = 0
    stall = 0
    max_stall = 15000
    while time.time() - start < max_time and it < 10000000:
        it += 1
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            continue
        
        delta = _swap_delta(cur, cand, in_set)
        if delta > 0:
            best_e += delta
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
            stall = 0
            if best_e == 76:
                return best, best_e
        else:
            stall += 1
            if stall > max_stall:
                break
    
    return best, best_e

def _try_improve_multi(pts, max_time=20.0, n_starts=4):
    """Run single-swap local search from multiple random perturbations."""
    rng_master = random.Random(999)
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    
    if best_e == 76:
        return best, best_e

    per_start = max_time / n_starts
    for s in range(n_starts):
        remaining = max_time - s * per_start
        if remaining < 2.0:
            break
        # Perturb: move a few random points to random nearby positions
        perturbed = list(best)
        in_set = set(perturbed)
        n_moves = min(n // 4, 8)
        for _ in range(n_moves):
            i = rng_master.randrange(n)
            cur = perturbed[i]
            a, b = cur
            da = rng_master.randint(-4, 4)
            db = rng_master.randint(-4, 4)
            cand = (a + da, b + db)
            if cand in in_set or cand == cur:
                continue
            if abs(cand[0]) > 64 or abs(cand[1]) > 64:
                continue
            in_set.discard(cur)
            in_set.add(cand)
            perturbed[i] = cand

        e_pert = _count_edges(perturbed)
        # Only keep perturbation if it's not too bad
        if e_pert < best_e - 8:
            continue

        start_t = time.time()
        res, e_res = _try_improve(perturbed, max_time=min(remaining - 1.0, per_start), seed=42 + s)
        if e_res > best_e:
            best, best_e = res, e_res
            if best_e == 76:
                return best, best_e

    return best, best_e

def _try_improve_2swap(pts, max_time=15.0):
    """Local search with 2-point swaps to escape local optima."""
    rng = random.Random(123)
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    pool = set()
    for (a,b) in best:
        for r in range(0, 10):
            for a2 in range(a-r, a+r+1):
                for b2 in range(b-r, b+r+1):
                    if _hexdist(a2,b2) <= 20:
                        pool.add((a2,b2))
    pool = list(pool)
    
    start = time.time()
    it = 0
    stall = 0
    max_stall = 30000
    while time.time() - start < max_time and it < 8000000:
        it += 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        cur1 = best[i]
        cur2 = best[j]
        cand1 = pool[rng.randrange(len(pool))]
        cand2 = pool[rng.randrange(len(pool))]
        if cand1 in in_set or cand2 in in_set or cand1 == cand2:
            continue
        if cand1 == cur1 or cand1 == cur2 or cand2 == cur1 or cand2 == cur2:
            continue
        
        s = in_set
        lost = _deg(cur1, s) + _deg(cur2, s)
        # Subtract 1 if cur1 and cur2 are neighbors
        if cur2 in s and any((cur2[0]+da, cur2[1]+db) == cur1 for da,db in _NEIGH):
            lost -= 1
        
        s2 = s - {cur1, cur2}
        gained = _deg(cand1, s2) + _deg(cand2, s2)
        a1,b1 = cand1
        a2,b2 = cand2
        # Add 1 if cand1 and cand2 are neighbors
        for da,db in _NEIGH:
            if (a1+da, b1+db) == cand2:
                gained += 1
                break
        
        delta = gained - lost
        if delta > 0:
            best_e += delta
            best[i] = cand1
            best[j] = cand2
            in_set.discard(cur1)
            in_set.discard(cur2)
            in_set.add(cand1)
            in_set.add(cand2)
            stall = 0
            if best_e == 76:
                return best, best_e
        else:
            stall += 1
            if stall > max_stall:
                break
    
    return best, best_e

def construct():
    deadline = time.time() + 140.0
    n = 32

    # Generate multiple initial candidates
    candidates = []
    
    # 1. Hexagonal Patch
    hex_patch = _build_hexagonal_patch(n)
    candidates.append(hex_patch)
    
    # 2. Spiral
    order = _spiral_order(12)
    spiral_pts = order[:n]
    candidates.append(spiral_pts)
    
    # 3. Greedy Construction with different seeds
    # Greedy is deterministic given the starting point and tie-breaking,
    # but we can vary the starting point or the candidate ordering.
    # For simplicity, we'll just run it once, but maybe with a different starting point.
    # Actually, the greedy method in Inspiration 1 used random seed to shuffle candidates?
    # No, it used random.seed but the logic was deterministic except for the initial point?
    # Let's just run a few variants.
    
    # Variant A: Standard Greedy (starts at 0,0)
    greedy_pts = _greedy_construct(n, seed=1)
    candidates.append(greedy_pts)
    
    # Variant B: Greedy starting at (1,0)
    # We can modify _greedy_construct to take a start point, but for now,
    # let's just trust the diversity of Hex and Spiral.
    # If we want more diversity, we can perturb the hex patch slightly.
    
    # Select the best initial candidate
    best = None
    best_e = -1
    for pts in candidates:
        if len(pts) != n:
            continue
        e = _count_edges(pts)
        if e > best_e:
            best_e = e
            best = pts

    if best_e == 76:
        return [[a, b] for (a, b) in best]

    # Phase 1: single-swap local search
    remaining = deadline - time.time()
    if remaining > 5.0:
        best1, e1 = _try_improve(best, max_time=min(remaining - 2.0, 15.0), seed=42)
        if e1 > best_e:
            best, best_e = best1, e1
            if best_e == 76:
                return [[a, b] for (a, b) in best]

    # Phase 2: 2-swap search
    remaining = deadline - time.time()
    if remaining > 5.0:
        best2, e2 = _try_improve_2swap(best, max_time=min(remaining - 2.0, 15.0))
        if e2 > best_e:
            best, best_e = best2, e2
            if best_e == 76:
                return [[a, b] for (a, b) in best]

    # Phase 3: multi-start local search (perturb + refine)
    remaining = deadline - time.time()
    if remaining > 8.0:
        best3, e3 = _try_improve_multi(best, max_time=min(remaining - 2.0, 25.0), n_starts=6)
        if e3 > best_e:
            best, best_e = best3, e3
            if best_e == 76:
                return [[a, b] for (a, b) in best]

    # Phase 4: final single-swap polish
    remaining = deadline - time.time()
    if remaining > 3.0:
        best4, e4 = _try_improve(best, max_time=min(remaining - 1.0, 10.0), seed=777)
        if e4 > best_e:
            best, best_e = best4, e4
            if best_e == 76:
                return [[a, b] for (a, b) in best]

    return [[a, b] for (a, b) in best]
