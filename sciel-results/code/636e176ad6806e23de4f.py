import math
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

def _dist_sq(a1, b1, a2, b2):
    da, db = a1 - a2, b1 - b2
    return da * da + da * db + db * db

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n = 32
    # Use hexagonal spiral construction which is known to be optimal
    # for this problem size (attains Harborth's bound).
    R = 3
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
    # Take the first n points
    best = [(a,b) for hd, ang, a, b in cands[:n]]
    best_set = set(best)
    
    # Verify initial edge count
    initial_edges = _count_edges(best)
    
    # Generate candidate points for local search (slightly larger region)
    R2 = 5
    all_candidates = []
    for a in range(-R2, R2+1):
        for b in range(-R2, R2+1):
            if _hexdist(a,b) <= R2:
                all_candidates.append((a, b))
    
    # Local improvement: try swapping points to increase edge count
    improved = True
    iteration = 0
    max_iterations = 200
    
    while improved and iteration < max_iterations:
        if time.time() > deadline - 10:
            break
        improved = False
        iteration += 1
        
        for i in range(len(best)):
            if time.time() > deadline - 10:
                break
            old_pt = best[i]
            
            for new_pt in all_candidates:
                if new_pt == old_pt or new_pt in best_set:
                    continue
                
                # Check minimum distance constraint: dist >= 1 from all other placed points
                valid = True
                for j in range(len(best)):
                    if i == j:
                        continue
                    if _dist_sq(new_pt[0], new_pt[1], best[j][0], best[j][1]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute new edge count if we swap
                # Remove old_pt, add new_pt
                new_set = set(best)
                new_set.remove(old_pt)
                new_set.add(new_pt)
                new_edges = _count_edges(new_set)
                
                if new_edges > initial_edges:
                    # Perform the swap
                    best[i] = new_pt
                    best_set.remove(old_pt)
                    best_set.add(new_pt)
                    initial_edges = new_edges
                    improved = True
                    break
            if improved:
                break
    
    # Final verification
    if len(best) != n:
        # Fallback to initial spiral
        best = [(a,b) for hd, ang, a, b in cands[:n]]
    
    return [[a,b] for (a,b) in best]
