import math
import itertools
import random
import time

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
    
    # Generate points in spiral order (by hex distance, then angle)
    # We need enough points to pick 32. Radius 3 gives 37, which is enough.
    R = 4
    pool = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if _hexdist(a,b) <= R:
                pool.append((a,b))
                
    # Sort by hex distance, then angle
    def _sort_key(p):
        a, b = p
        hd = _hexdist(a,b)
        x = a + 0.5*b
        y = 0.8660254037844386*b
        ang = math.atan2(y, x)
        return (hd, ang)
        
    pool.sort(key=_sort_key)
    
    # Take the first 32 points
    best = pool[:n]
    best_e = _count_edges(best)
    in_set = set(best)
    
    # If we haven't reached the optimum, run a local search to improve
    if best_e < 76:
        deadline = time.time() + 110 # Safety margin
        rng = random.Random(0)
        
        it = 0
        # Use incremental edge count updates for speed
        # To do this efficiently, we need to know the current edge count.
        # When replacing point P_old with P_new:
        # 1. Remove edges connected to P_old that are still in the set.
        # 2. Add edges connected to P_new that are in the set.
        
        while time.time() < deadline and it < 2000000:
            it += 1
            i = rng.randrange(n)
            cur = best[i]
            cand = pool[rng.randrange(len(pool))]
            if cand == cur or cand in in_set:
                continue
            
            # Calculate delta in edge count
            # Edges lost: neighbors of cur that are in in_set (excluding cand, which isn't in set yet, but cur is)
            # Note: in_set contains cur.
            # We are removing cur from in_set and adding cand.
            
            # Count edges from cur to other points in in_set
            edges_lost = 0
            for da, db in _NEIGH:
                if (cur[0]+da, cur[1]+db) in in_set:
                    edges_lost += 1
            
            # Count edges from cand to other points in in_set (excluding cur, which is being removed)
            # Since cand is not in in_set, we just check neighbors in in_set.
            # But we are removing cur. So if cand is a neighbor of cur, that edge is not "lost" in the sense of disappearing from the graph, 
            # but cur is gone. Wait.
            # The set of points changes from S to S \ {cur} U {cand}.
            # Edges in S: E(S).
            # Edges in S': E(S').
            # E(S') = E(S) - (edges incident to cur) + (edges incident to cand in S')
            # Edges incident to cur in S: count of neighbors of cur in S.
            # Edges incident to cand in S': count of neighbors of cand in S'.
            # S' = (S \ {cur}) U {cand}.
            # Neighbors of cand in S' are neighbors of cand in (S \ {cur}) plus cand itself? No, edges are between distinct points.
            # So neighbors of cand in S' are {v in S \ {cur} | dist(cand, v) == 1}.
            
            edges_gained = 0
            for da, db in _NEIGH:
                nb = (cand[0]+da, cand[1]+db)
                if nb in in_set and nb != cur:
                    edges_gained += 1
            
            delta = edges_gained - edges_lost
            
            if delta > 0:
                # Accept move
                best_e += delta
                best[i] = cand
                in_set.discard(cur)
                in_set.add(cand)
                if best_e == 76:
                    break
            
            # Optional: Simulated Annealing accept step (delta < 0) to escape local optima
            # For now, stick to hill climbing as it's faster and 76 is the global max.
            # If hill climbing fails, we might need SA.
        
    return [[a,b] for (a,b) in best]
