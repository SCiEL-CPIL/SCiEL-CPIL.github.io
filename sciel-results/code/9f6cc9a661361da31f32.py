"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 84.0
    
    n = 128
    # Precompute all 4-term arithmetic progressions
    aps = []
    for a in range(n):
        for d in range(1, n):
            e = a + 3*d
            if e < n:
                aps.append((a, a+d, a+2*d, e))
    
    n_aps = len(aps)
    
    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[e].append(idx)
    
    def distinct_count(ca, cb, cc, ce):
        if ca == cb:
            if ca == cc:
                if ca == ce:
                    return 1
                return 2
            else:
                if ce == ca:
                    return 2
                elif ce == cc:
                    return 2
                else:
                    return 3
        else:
            if cc == ca:
                if ce == ca or ce == cb:
                    return 2
                else:
                    return 3
            elif cc == cb:
                if ce == ca or ce == cb:
                    return 2
                else:
                    return 3
            else:
                if ce == ca or ce == cb or ce == cc:
                    return 3
                else:
                    return 4
    
    def compute_violations(colors):
        viol_count = 0
        ap_distinct = [0] * n_aps
        for idx in range(n_aps):
            a, b, c, e = aps[idx]
            dc = distinct_count(colors[a], colors[b], colors[c], colors[e])
            ap_distinct[idx] = dc
            if dc <= 2:
                viol_count += 1
        return viol_count, ap_distinct
    
    def local_search(colors, k, max_iters=20000):
        colors = list(colors)
        viol_count, ap_distinct = compute_violations(colors)
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            if viol_count == 0:
                break
            
            improved = False
            # Try a few random moves
            for _ in range(10):
                pos = random.randint(0, n-1)
                old_color = colors[pos]
                new_color = random.randint(0, k-1)
                if new_color == old_color:
                    continue
                
                delta = 0
                for ap_idx in pos_aps[pos]:
                    a, b, c, e = aps[ap_idx]
                    old_dc = ap_distinct[ap_idx]
                    ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
                    if a == pos:
                        ca = new_color
                    elif b == pos:
                        cb = new_color
                    elif c == pos:
                        cc = new_color
                    else:
                        ce = new_color
                    new_dc = distinct_count(ca, cb, cc, ce)
                    old_v = 1 if old_dc <= 2 else 0
                    new_v = 1 if new_dc <= 2 else 0
                    delta += new_v - old_v
                
                if delta <= 0:
                    colors[pos] = new_color
                    for ap_idx in pos_aps[pos]:
                        a, b, c, e = aps[ap_idx]
                        ap_distinct[ap_idx] = distinct_count(colors[a], colors[b], colors[c], colors[e])
                    viol_count += delta
                    improved = True
                    break
            
            if not improved and it % 100 == 0:
                # Perturbation
                for _ in range(3):
                    pos = random.randint(0, n-1)
                    old_color = colors[pos]
                    new_color = random.randint(0, k-1)
                    if new_color != old_color:
                        delta = 0
                        for ap_idx in pos_aps[pos]:
                            a, b, c, e = aps[ap_idx]
                            old_dc = ap_distinct[ap_idx]
                            ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
                            if a == pos:
                                ca = new_color
                            elif b == pos:
                                cb = new_color
                            elif c == pos:
                                cc = new_color
                            else:
                                ce = new_color
                            new_dc = distinct_count(ca, cb, cc, ce)
                            old_v = 1 if old_dc <= 2 else 0
                            new_v = 1 if new_dc <= 2 else 0
                            delta += new_v - old_v
                        colors[pos] = new_color
                        for ap_idx in pos_aps[pos]:
                            a, b, c, e = aps[ap_idx]
                            ap_distinct[ap_idx] = distinct_count(colors[a], colors[b], colors[c], colors[e])
                        viol_count += delta
                        improved = True
                        break
                if not improved:
                    break
        
        return colors, viol_count
    
    def try_merge(colors, k):
        colors = list(colors)
        color_pos = [set() for _ in range(k)]
        for i, c in enumerate(colors):
            color_pos[c].add(i)
        
        for c1 in range(k):
            if not color_pos[c1]:
                continue
            for c2 in range(c1+1, k):
                if not color_pos[c2]:
                    continue
                affected = list(color_pos[c2])
                affected_aps = set()
                for pos in affected:
                    affected_aps.update(pos_aps[pos])
                
                test_colors = colors[:]
                for pos in affected:
                    test_colors[pos] = c1
                
                valid = True
                for ap_idx in affected_aps:
                    a, b, c, e = aps[ap_idx]
                    if distinct_count(test_colors[a], test_colors[b], test_colors[c], test_colors[e]) <= 2:
                        valid = False
                        break
                
                if valid:
                    return test_colors, k-1
        return None, k
    
    best_result = None
    best_num_colors = float('inf')
    
    for k in range(3, 7):
        if time.time() - start_time > DEADLINE:
            break
        
        for restart in range(10):
            if time.time() - start_time > DEADLINE:
                break
            
            colors = [random.randint(0, k-1) for _ in range(n)]
            for i in range(min(k, n)):
                colors[i] = i % k
            
            colors, viol = local_search(colors, k, max_iters=15000)
            
            if viol == 0:
                current_k = k
                for _ in range(10):
                    if time.time() - start_time > DEADLINE:
                        break
                    merged, new_k = try_merge(colors, current_k)
                    if merged is not None:
                        colors = merged
                        current_k = new_k
                    else:
                        break
                
                actual = len(set(colors))
                if actual < best_num_colors:
                    best_num_colors = actual
                    best_result = colors[:]
                    if actual == 3:
                        return best_result
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
