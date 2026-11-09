def construct():
    import time, random
    import numpy as np
    t0 = time.time()
    DEADLINE = t0 + 85.0
    n = 128
    aps = []
    for a in range(1, n+1):
        md = (n - a)//3
        for d in range(1, md+1):
            aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    m = len(aps)
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    
    ap_arr = np.array(aps, dtype=np.int32) # shape (m, 4)
    
    def try_k(k, budget):
        end = t0 + budget
        best = None
        best_du = 10**9
        rng = random
        
        # Precompute bad lookup table for this k
        bad_lookup = np.zeros((k, k, k, k), dtype=bool)
        for a in range(k):
            for b in range(k):
                for c in range(k):
                    for d in range(k):
                        vals = {a, b, c, d}
                        if len(vals) < 3:
                            bad_lookup[a, b, c, d] = True
        
        while time.time() < end:
            # Initialize colors
            if rng.random() < 0.5:
                colors = np.array([i % k for i in range(n)], dtype=np.int32)
            else:
                colors = np.array([rng.randrange(k) for _ in range(n)], dtype=np.int32)
            
            # Compute initial ap_bad correctly using colors
            col_vals = colors[ap_arr]
            ap_bad = bad_lookup[col_vals[:,0], col_vals[:,1], col_vals[:,2], col_vals[:,3]]
            
            vset = set(np.where(ap_bad)[0].tolist())
            
            stall = 0
            while time.time() < end and vset:
                # Pick a violating AP
                ei = rng.choice(list(vset))
                ap = aps[ei]
                p0, p1, p2, p3 = ap
                
                # Candidate moves
                best_delta = 10**9
                best_mv = None
                
                for p in (p0, p1, p2, p3):
                    cur_c = colors[p]
                    for c in range(k):
                        if c == cur_c:
                            continue
                        
                        delta = 0
                        ap_indices = pos_aps[p]
                        
                        for idx in ap_indices:
                            a0, a1, a2, a3 = aps[idx]
                            c0, c1, c2, c3 = colors[a0], colors[a1], colors[a2], colors[a3]
                            
                            was_bad = bool(ap_bad[idx])
                            
                            if a0 == p:
                                nc0, nc1, nc2, nc3 = c, c1, c2, c3
                            elif a1 == p:
                                nc0, nc1, nc2, nc3 = c0, c, c2, c3
                            elif a2 == p:
                                nc0, nc1, nc2, nc3 = c0, c1, c, c3
                            else:
                                nc0, nc1, nc2, nc3 = c0, c1, c2, c
                            
                            is_bad_now = bool(bad_lookup[nc0, nc1, nc2, nc3])
                            
                            if was_bad and not is_bad_now:
                                delta -= 1
                            elif not was_bad and is_bad_now:
                                delta += 1
                            
                        if delta < best_delta:
                            best_delta = delta
                            best_mv = (p, c)
                
                if best_mv is None:
                    break
                    
                p, c = best_mv
                colors[p] = c
                
                # Update ap_bad and vset
                for idx in pos_aps[p]:
                    a0, a1, a2, a3 = aps[idx]
                    c0, c1, c2, c3 = colors[a0], colors[a1], colors[a2], colors[a3]
                    is_bad_now = bool(bad_lookup[c0, c1, c2, c3])
                    
                    if is_bad_now and not ap_bad[idx]:
                        vset.add(idx)
                    elif not is_bad_now and ap_bad[idx]:
                        vset.discard(idx)
                    
                    ap_bad[idx] = is_bad_now
                
                if best_delta < 0:
                    stall = 0
                else:
                    stall += 1
                    if stall > 50:
                        p2 = rng.randrange(n)
                        c2 = rng.randrange(k)
                        if c2 != colors[p2]:
                            colors[p2] = c2
                            for idx in pos_aps[p2]:
                                a0, a1, a2, a3 = aps[idx]
                                c0, c1, c2, c3 = colors[a0], colors[a1], colors[a2], colors[a3]
                                is_bad_now = bool(bad_lookup[c0, c1, c2, c3])
                                
                                if is_bad_now and not ap_bad[idx]:
                                    vset.add(idx)
                                elif not is_bad_now and ap_bad[idx]:
                                    vset.discard(idx)
                                
                                ap_bad[idx] = is_bad_now
                        stall = 0
            
            if not vset:
                du = len(set(colors))
                if du < best_du:
                    best_du = du
                    best = [int(x) for x in colors]
                if best_du == k and k == 3:
                    return best
        return best

    result = None
    if time.time() < DEADLINE - 5:
        res = try_k(3, min(20, DEADLINE - time.time() - 5))
        if res is not None:
            ok = True
            for ap in aps:
                if len({res[ap[0]], res[ap[1]], res[ap[2]], res[ap[3]]}) < 3:
                    ok = False
                    break
            if ok:
                result = res
    
    if result is None and time.time() < DEADLINE - 5:
        res = try_k(4, min(20, DEADLINE - time.time() - 5))
        if res is not None:
            ok = True
            for ap in aps:
                if len({res[ap[0]], res[ap[1]], res[ap[2]], res[ap[3]]}) < 3:
                    ok = False
                    break
            if ok:
                result = res
                
    if result is None:
        for k in [5, 6, 7, 8]:
            if time.time() > DEADLINE - 2:
                break
            res = try_k(k, min(5, DEADLINE - time.time() - 2))
            if res is not None:
                ok = True
                for ap in aps:
                    if len({res[ap[0]], res[ap[1]], res[ap[2]], res[ap[3]]}) < 3:
                        ok = False
                        break
                if ok:
                    result = res
                    break

    if result is None:
        result = list(range(128))
        
    return result
