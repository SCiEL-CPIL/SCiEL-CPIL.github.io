def construct():
    import time
    import numpy as np
    
    t0 = time.time()
    DEADLINE = t0 + 80.0
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    m = len(aps)
    ap_arr = np.array(aps, dtype=np.int32)
    
    # For each position, which APs does it belong to?
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    
    def count_distinct_4(c0, c1, c2, c3):
        """Count distinct colors in 4 values."""
        cnt = 1
        if c1 != c0: cnt += 1
        if c2 != c0 and c2 != c1: cnt += 1
        if c3 != c0 and c3 != c1 and c3 != c2: cnt += 1
        return cnt
    
    def try_find_valid(k, time_budget):
        """Try to find a valid k-coloring using local search."""
        end = t0 + time_budget
        
        for attempt in range(100):
            if time.time() > end:
                break
            
            colors = np.random.randint(0, k, size=n, dtype=np.int32)
            
            # Compute initial violation count
            dist = np.zeros(m, dtype=np.int32)
            for i in range(m):
                ap = ap_arr[i]
                dist[i] = count_distinct_4(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
            
            v = np.sum(dist < 3)
            if v == 0:
                return colors
            
            stall = 0
            for step in range(10000):
                if time.time() > end:
                    break
                if v == 0:
                    return colors
                
                # Pick a random violating AP
                viol_idx = np.where(dist < 3)[0]
                if len(viol_idx) == 0:
                    return colors
                idx = viol_idx[np.random.randint(len(viol_idx))]
                ap = ap_arr[idx]
                
                # Try changing one position in this AP
                best_delta = 0
                best_move = None
                
                for pos in ap:
                    old_c = colors[pos]
                    for new_c in range(k):
                        if new_c == old_c:
                            continue
                        
                        # Compute delta: how many APs change from violating to valid or vice versa
                        delta = 0
                        for aidx in pos_aps[pos]:
                            a2 = ap_arr[aidx]
                            old_d = dist[aidx]
                            # Compute new distinct count
                            vals = [colors[a2[0]], colors[a2[1]], colors[a2[2]], colors[a2[3]]]
                            for j, p in enumerate(a2):
                                if p == pos:
                                    vals[j] = new_c
                            new_d = count_distinct_4(vals[0], vals[1], vals[2], vals[3])
                            was_viol = old_d < 3
                            now_viol = new_d < 3
                            if was_viol and not now_viol:
                                delta -= 1
                            elif (not was_viol) and now_viol:
                                delta += 1
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (pos, new_c)
                
                if best_move is None:
                    # Random kick
                    pos = np.random.randint(n)
                    new_c = np.random.randint(k)
                    if new_c == colors[pos]:
                        continue
                    old_c = colors[pos]
                    colors[pos] = new_c
                    for aidx in pos_aps[pos]:
                        a2 = ap_arr[aidx]
                        dist[aidx] = count_distinct_4(colors[a2[0]], colors[a2[1]], colors[a2[2]], colors[a2[3]])
                    v = np.sum(dist < 3)
                    stall += 1
                else:
                    pos, new_c = best_move
                    colors[pos] = new_c
                    for aidx in pos_aps[pos]:
                        a2 = ap_arr[aidx]
                        dist[aidx] = count_distinct_4(colors[a2[0]], colors[a2[1]], colors[a2[2]], colors[a2[3]])
                    v = np.sum(dist < 3)
                    if best_delta < 0:
                        stall = 0
                    else:
                        stall += 1
                
                if stall > 300:
                    break
        
        return None
    
    result = None
    
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > DEADLINE - 5:
            break
        budget = min(12, DEADLINE - 5 - (time.time() - t0))
        if budget < 1:
            break
        
        colors = try_find_valid(k, budget)
        if colors is not None:
            # Validate
            ok = True
            for ap in aps:
                if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                    ok = False
                    break
            if ok:
                result = colors
                break
    
    if result is None:
        # Fallback
        result = np.array([i % 5 for i in range(n)], dtype=np.int32)
    
    return result.tolist()
