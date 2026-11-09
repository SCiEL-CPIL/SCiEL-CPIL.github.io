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
    deadline = 140.0  # Safety margin
    
    n = 32
    
    # Build a good initial configuration using spiral order
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
    
    # Candidate pool for swaps - larger pool
    R2 = 6
    all_candidates = []
    for a in range(-R2, R2+1):
        for b in range(-R2, R2+1):
            if _hexdist(a,b) <= R2:
                all_candidates.append((a, b))
    
    # Local Improvement with incremental edge counting
    improved = True
    iteration = 0
    max_iterations = 20000
    
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
                edges_lost = 0
                for da, db in _NEIGH:
                    if (old_pt[0]+da, old_pt[1]+db) in best_set:
                        edges_lost += 1
                        
                edges_gained = 0
                for da, db in _NEIGH:
                    nb = (new_pt[0]+da, new_pt[1]+db)
                    if nb in best_set and nb != old_pt:
                        edges_gained += 1
                        
                delta = edges_gained - edges_lost
                if delta > 0:
                    best[i] = new_pt
                    best_set.remove(old_pt)
                    best_set.add(new_pt)
                    current_edges += delta
                    improved = True
                    break
            if improved:
                break
    
    # Second pass: try 2-swap moves to escape local optima
    if time.time() < deadline:
        improved2 = True
        iter2 = 0
        max_iter2 = 5000
        while improved2 and iter2 < max_iter2:
            if time.time() > deadline:
                break
            improved2 = False
            iter2 += 1
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(best)):
                    if time.time() > deadline:
                        break
                    old1, old2 = best[i], best[j]
                    for new1 in all_candidates:
                        if new1 in best_set or new1 == old1:
                            continue
                        for new2 in all_candidates:
                            if new2 in best_set or new2 == old2 or new2 == new1:
                                continue
                            # Check distance constraints
                            valid = True
                            for k in range(len(best)):
                                if k == i or k == j:
                                    continue
                                if _dist_sq(new1[0], new1[1], best[k][0], best[k][1]) < 1:
                                    valid = False
                                    break
                                if _dist_sq(new2[0], new2[1], best[k][0], best[k][1]) < 1:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            if _dist_sq(new1[0], new1[1], new2[0], new2[1]) < 1:
                                continue
                            
                            # Edge change calculation
                            # Remove old1, old2
                            lost = 0
                            for da, db in _NEIGH:
                                if (old1[0]+da, old1[1]+db) in best_set:
                                    lost += 1
                                if (old2[0]+da, old2[1]+db) in best_set:
                                    lost += 1
                            # Add new1, new2
                            gained = 0
                            for da, db in _NEIGH:
                                nb1 = (new1[0]+da, new1[1]+db)
                                if nb1 in best_set and nb1 != old1 and nb1 != old2:
                                    gained += 1
                                nb2 = (new2[0]+da, new2[1]+db)
                                if nb2 in best_set and nb2 != old1 and nb2 != old2:
                                    gained += 1
                            # If new1 and new2 are neighbors, that's an extra edge
                            if _dist_sq(new1[0], new1[1], new2[0], new2[1]) == 1:
                                gained += 1
                            
                            delta2 = gained - lost
                            if delta2 > 0:
                                best[i] = new1
                                best[j] = new2
                                best_set.remove(old1)
                                best_set.remove(old2)
                                best_set.add(new1)
                                best_set.add(new2)
                                current_edges += delta2
                                improved2 = True
                                break
                        if improved2:
                            break
                    if improved2:
                        break
                if improved2:
                    break
    
    # Final verification
    if len(best) != n:
        best = [(a,b) for hd, ang, a, b in cands[:n]]
        
    return [[a,b] for (a,b) in best]
