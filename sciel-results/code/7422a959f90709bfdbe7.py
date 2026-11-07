import math
import time

def construct():
    N = 256
    deadline = time.time() + 85.0
    
    weights = list(range(1, N + 1))
    
    # Build conflict graph (edges where gcd > 1)
    conflict = [0] * N
    for i in range(N):
        vi = i + 1
        for j in range(i + 1, N):
            vj = j + 1
            if math.gcd(vi, vj) > 1:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Greedy initial solution for a strong lower bound
    def greedy():
        picked = set()
        total = 0
        avail = set(range(N))
        while avail:
            # Pick highest weight vertex that doesn't conflict with picked
            best = None
            best_w = -1
            for v in avail:
                if weights[v] > best_w:
                    ok = True
                    for p in picked:
                        if conflict[v] & (1 << p):
                            ok = False
                            break
                    if ok:
                        best = v
                        best_w = weights[v]
            if best is None:
                break
            picked.add(best)
            total += weights[best]
            avail.discard(best)
        return total, list(picked)
    
    best_sum, best_vertices = greedy()
    
    # Coloring bound for B&B
    def coloring_bound(candidates_mask):
        remaining = candidates_mask
        total = 0
        while remaining:
            color_max = 0
            rem = remaining
            color_set = 0
            while rem:
                best_v = -1
                best_w = -1
                m = rem
                while m:
                    v = (m & -m).bit_length() - 1
                    if weights[v] > best_w:
                        best_w = weights[v]
                        best_v = v
                    m &= m - 1
                if best_v < 0:
                    break
                color_set |= (1 << best_v)
                if best_w > color_max:
                    color_max = best_w
                rem &= ~(1 << best_v)
                rem &= ~conflict[best_v]
            total += color_max
            remaining &= ~color_set
        return total
    
    import sys
    sys.setrecursionlimit(100000)
    
    best = [best_sum]
    best_set = [best_vertices]
    
    def bb(candidates_mask, picked_mask, current_sum):
        if time.time() > deadline:
            return
        if not candidates_mask:
            if current_sum > best[0]:
                best[0] = current_sum
                best_set[0] = list(picked_mask)
            return
        
        # Pick highest weight vertex in candidates
        best_v = -1
        best_w = -1
        m = candidates_mask
        while m:
            v = (m & -m).bit_length() - 1
            if weights[v] > best_w:
                best_w = weights[v]
                best_v = v
            m &= m - 1
        
        if best_v < 0:
            return
        
        bound = current_sum + coloring_bound(candidates_mask)
        if bound <= best[0]:
            return
        
        # Include best_v
        new_cand = candidates_mask & ~(1 << best_v) & ~conflict[best_v]
        bb(new_cand, picked_mask | (1 << best_v), current_sum + weights[best_v])
        
        # Exclude best_v
        new_cand2 = candidates_mask & ~(1 << best_v)
        bb(new_cand2, picked_mask, current_sum)
    
    full_mask = (1 << N) - 1
    bb(full_mask, 0, 0)
    
    result = sorted([v + 1 for v in best_set[0]])
    if not result:
        result = [1]
    return result
