import math
import time

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _dist_sq(a1, b1, a2, b2):
    da, db = a1 - a2, b1 - b2
    return da * da + da * db + db * db

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n = 32
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
    best = [(a,b) for hd, ang, a, b in cands[:n]]
    best_set = set(best)
    current_edges = _count_edges(best)
    
    # Candidate pool for swaps
    R2 = 5
    all_candidates = []
    for a in range(-R2, R2+1):
        for b in range(-R2, R2+1):
            if _hexdist(a,b) <= R2:
                all_candidates.append((a, b))
    
    # Local Improvement with incremental edge counting
    improved = True
    iteration = 0
    max_iterations = 5000  # Can be higher due to speedup
    
    while improved and iteration < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iteration += 1
        
        for i in range(len(best)):
            if time.time() > deadline:
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
                
                # Calculate edge change
                # Remove old_pt: lose edges from old_pt to its neighbors in best_set
                edges_lost = 0
                for da, db in _NEIGH:
                    if (old_pt[0]+da, old_pt[1]+db) in best_set:
                        edges_lost += 1
                        
                # Add new_pt: gain edges from new_pt to its neighbors in best_set (excluding old_pt)
                edges_gained = 0
                for da, db in _NEIGH:
                    nb = (new_pt[0]+da, new_pt[1]+db)
                    if nb in best_set and nb != old_pt:
                        edges_gained += 1
                        
                delta = edges_gained - edges_lost
                if delta > 0:
                    # Perform swap
                    best[i] = new_pt
                    best_set.remove(old_pt)
                    best_set.add(new_pt)
                    current_edges += delta
                    improved = True
                    break
            if improved:
                break
    
    # Final verification
    if len(best) != n:
        best = [(a,b) for hd, ang, a, b in cands[:n]]
        
    return [[a,b] for (a,b) in best]
