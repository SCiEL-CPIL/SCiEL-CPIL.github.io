import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    
    # Precompute all 4-term APs (0-indexed) and index by position
    aps = []
    ap_index = {}
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                idx = len(aps)
                aps.append((a, b, c, e))
                for pos in (a, b, c, e):
                    if pos not in ap_index:
                        ap_index[pos] = []
                    ap_index[pos].append(idx)
    
    num_aps = len(aps)
    
    def ap_violation(coloring, idx):
        a, b, c, e = aps[idx]
        return len({coloring[a], coloring[b], coloring[c], coloring[e]}) < 3
    
    # Try k=3,4,5,...
    for k in range(3, 130):
        if time.time() > deadline:
            break
        
        for restart in range(30):
            if time.time() > deadline:
                break
            
            coloring = [random.randint(0, k-1) for _ in range(n)]
            viol = sum(1 for i in range(num_aps) if ap_violation(coloring, i))
            
            if viol == 0:
                used = sorted(set(coloring))
                cmap = {c:i for i,c in enumerate(used)}
                return [cmap[c] for c in coloring]
            
            # Simulated annealing with incremental updates
            temp = 100.0
            for step in range(20000):
                if time.time() > deadline:
                    break
                
                if viol == 0:
                    used = sorted(set(coloring))
                    cmap = {c:i for i,c in enumerate(used)}
                    return [cmap[c] for c in coloring]
                
                pos = random.randint(0, n-1)
                old = coloring[pos]
                new = random.randint(0, k-1)
                if new == old:
                    continue
                
                # Compute change in violations
                delta = 0
                for idx in ap_index.get(pos, []):
                    was_viol = ap_violation(coloring, idx)
                    coloring[pos] = new
                    is_viol = ap_violation(coloring, idx)
                    if was_viol and not is_viol:
                        delta -= 1
                    elif not was_viol and is_viol:
                        delta += 1
                    coloring[pos] = old
                
                if delta < 0 or random.random() < pow(2.718, -delta/temp):
                    coloring[pos] = new
                    viol += delta
                else:
                    coloring[pos] = old
                
                temp *= 0.9995
    
    # Fallback: use all distinct colors (always valid)
    return list(range(n))
