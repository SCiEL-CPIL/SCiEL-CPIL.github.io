import random
import time

def construct():
    # Precompute 4-APs
    aps = []
    for d in range(1, 33): 
        for start in range(1, 128 - 3*d + 1):
            aps.append((start, start+d, start+2*d, start+3*d))
            
    aps_by_last = {i: [] for i in range(1, 129)}
    for ap in aps:
        last = ap[3]
        aps_by_last[last].append(ap)
        
    colors = [None] * 129 # 1-indexed
    
    def backtrack_k(pos, k):
        if pos > 128:
            return True
        
        color_order = list(range(k))
        random.shuffle(color_order)
        
        for c in color_order:
            colors[pos] = c
            
            valid = True
            for ap in aps_by_last[pos]:
                a, b, d_idx, e = ap
                # e is pos
                ca = colors[a]
                cb = colors[b]
                cd = colors[d_idx]
                ce = colors[e]
                # All must be assigned (they are, since a,b,d < pos)
                distinct = len({ca, cb, cd, ce})
                if distinct < 3:
                    valid = False
                    break
            
            if valid:
                if backtrack_k(pos + 1, k):
                    return True
            
            colors[pos] = None
        
        return False

    # Try 3 colors
    start_time = time.time()
    limit = 5.0
    while time.time() - start_time < limit:
        for i in range(1, 129):
            colors[i] = None
        if backtrack_k(1, 3):
            # Verify
            ok = True
            for ap in aps:
                c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if len({c1, c2, c3, c4}) < 3:
                    ok = False
                    break
            if ok:
                return [colors[i] for i in range(1, 129)]
        
    # Try 4 colors
    start_time = time.time()
    limit = 5.0
    while time.time() - start_time < limit:
        for i in range(1, 129):
            colors[i] = None
        if backtrack_k(1, 4):
            # Verify
            ok = True
            for ap in aps:
                c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if len({c1, c2, c3, c4}) < 3:
                    ok = False
                    break
            if ok:
                return [colors[i] for i in range(1, 129)]
                
    # Fallback: Greedy 4-coloring
    colors = [0] * 129
    for x in range(1, 129):
        for c in range(4):
            ok = True
            for ap in aps_by_last[x]:
                a, b, d_idx, e = ap
                ca = colors[a]
                cb = colors[b]
                cd = colors[d_idx]
                if len({ca, cb, cd, c}) < 3:
                    ok = False
                    break
            if ok:
                colors[x] = c
                break
                
    return [colors[i] for i in range(1, 129)]

