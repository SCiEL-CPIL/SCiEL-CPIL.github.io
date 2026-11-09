import math
import itertools

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

def construct():
    n = 32
    # Build hexagon radius 3
    R1 = 3
    hex1 = []
    for a in range(-R1, R1+1):
        for b in range(-R1, R1+1):
            if _hexdist(a,b) <= R1:
                hex1.append((a,b))
    
    # Should be 37
    assert len(hex1) == 37, f"Expected 37, got {len(hex1)}"
    
    # Precompute degrees and adjacency
    hex_set = set(hex1)
    deg = {}
    adj = {}
    for p in hex1:
        neighbors = []
        for da, db in _NEIGH:
            np_ = (p[0]+da, p[1]+db)
            if np_ in hex_set:
                neighbors.append(np_)
        deg[p] = len(neighbors)
        adj[p] = set(neighbors)
        
    # Total edges in full hexagon
    E_full = sum(deg.values()) // 2
    
    # We want to remove 5 points to maximize edges in the remaining 32.
    # E_kept = E_full - E(removed, kept) - E(removed, removed)
    # Let R be removed set.
    # Sum(deg[p] for p in R) = E(R, K) + 2*E(R, R)
    # So E(R, K) = Sum(deg[p] for p in R) - 2*E(R, R)
    # E_kept = E_full - (Sum(deg[p] for p in R) - 2*E(R, R)) - E(R, R)
    # E_kept = E_full - Sum(deg[p] for p in R) + E(R, R)
    
    best_e = -1
    best_keep = None
    best_remove = None
    
    # Map point to index for faster lookup if needed, but set is fine
    # We iterate over combinations of indices 0..36
    # To speed up, we can precompute the "cost" of removing a single point?
    # No, E(R, R) depends on interactions.
    
    # Optimization: Iterate combinations.
    # 435897 combinations.
    # For each, calculate sum_deg and E_RR.
    
    # To make E_RR calculation fast, we can just check pairs in the small remove set.
    # remove set has size 5. Pairs = 10.
    
    for remove_idx in itertools.combinations(range(37), 5):
        # Calculate sum of degrees
        sum_deg = 0
        remove_pts = []
        for idx in remove_idx:
            p = hex1[idx]
            sum_deg += deg[p]
            remove_pts.append(p)
            
        # Calculate E(R, R)
        e_rr = 0
        for i in range(5):
            for j in range(i+1, 5):
                if remove_pts[j] in adj[remove_pts[i]]:
                    e_rr += 1
                    
        e_kept = E_full - sum_deg + e_rr
        
        if e_kept > best_e:
            best_e = e_kept
            best_remove = remove_idx
            # Construct keep set for verification or early exit
            # We can construct it now to check if it's valid (it always is)
            # But we only need to return it at the end.
            # However, if we find 76, we can break.
            if best_e == 76:
                break
                
    if best_remove is not None:
        remove_set = set(best_remove)
        keep = [p for i, p in enumerate(hex1) if i not in remove_set]
        return [[a,b] for (a,b) in keep]
        
    # Fallback: If no 76 found (unlikely), use local search on larger pool
    import random
    import time
    deadline = time.time() + 100
    rng = random.Random(0)
    
    # Use hexagon radius 4 as pool
    R2 = 4
    pool = []
    for a in range(-R2, R2+1):
        for b in range(-R2, R2+1):
            if _hexdist(a,b) <= R2:
                pool.append((a,b))
        
    # Start from a spiral ordering
    def _spiral_order(limit):
        R = limit
        cands = []
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if _hexdist(a,b) <= R:
                    hd = _hexdist(a,b)
                    x = a + 0.5*b
                    y = 0.8660254037844386*b
                    ang = math.atan2(y, x)
                    cands.append((hd, ang, a, b))
        cands.sort()
        return [(a,b) for hd, ang, a, b in cands]
        
    order = _spiral_order(R2)
    best = list(order[:n])
    best_e = _count_edges(best)
    
    in_set = set(best)
    pool_list = list(pool)
    
    it = 0
    while time.time() < deadline and it < 500000:
        it += 1
        i = rng.randrange(n)
        cur = best[i]
        cand = pool_list[rng.randrange(len(pool_list))]
        if cand == cur or cand in in_set:
            continue
        trial = best[:i] + [cand] + best[i+1:]
        e2 = _count_edges(trial)
        if e2 > best_e:
            best_e = e2
            best = trial
            in_set.discard(cur)
            in_set.add(cand)
            
    return [[a,b] for (a,b) in best]
