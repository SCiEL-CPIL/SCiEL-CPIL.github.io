def construct():
    import time, random
    import numpy as np
    t0 = time.time()
    DEADLINE = t0 + 50.0
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
        # bad_lookup is a 4D array of bools
        # bad_lookup[a,b,c,d] is True if distinct count < 3
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
            
            # Initialize ap_bad
            # ap_bad[i] is True if AP i is bad
            # We can vectorize this? 
            # ap_arr is (m, 4). We need to index bad_lookup with these.
            # bad_lookup[ap_arr[:,0], ap_arr[:,1], ap_arr[:,2], ap_arr[:,3]]
            ap_bad = bad_lookup[ap_arr[:,0], ap_arr[:,1], ap_arr[:,2], ap_arr[:,3]]
            
            vset = set(np.where(ap_bad)[0])
            
            stall = 0
            while time.time() < end and vset:
                # Pick a violating AP
                ei = rng.choice(list(vset))
                ap = aps[ei]
                p0, p1, p2, p3 = ap
                
                # Candidate moves: change p0, p1, p2, or p3 to any color
                best_delta = 10**9
                best_mv = None
                
                # To evaluate a move (p, c):
                # 1. Find all APs containing p: pos_aps[p]
                # 2. For each such AP j, check if it is currently bad.
                # 3. Simulate changing p to c in AP j.
                # 4. Check if new state is bad.
                # 5. Delta = (new_bad_count - old_bad_count) for these APs.
                
                # We can optimize by precomputing the list of APs for each position.
                # pos_aps[p] is a list of indices.
                
                for p in (p0, p1, p2, p3):
                    cur_c = colors[p]
                    # Only try colors different from current? 
                    # If we try same color, delta is 0. We want min delta, usually negative.
                    # So skip cur_c.
                    for c in range(k):
                        if c == cur_c:
                            continue
                        
                        # Evaluate delta
                        delta = 0
                        # Get indices of APs containing p
                        ap_indices = pos_aps[p]
                        if not ap_indices:
                            continue
                        
                        # For each AP in ap_indices, determine if it was bad and if it is bad after change
                        # We need to know the other 3 colors in each AP.
                        # This requires iterating over ap_indices.
                        # Since len(ap_indices) is small (max ~10-20), this is fast.
                        
                        # Optimization: Precompute the "other 3 colors" for each position in each AP?
                        # No, just iterate.
                        
                        for idx in ap_indices:
                            # Get the 4 positions of this AP
                            a0, a1, a2, a3 = aps[idx]
                            # Get current colors
                            c0, c1, c2, c3 = colors[a0], colors[a1], colors[a2], colors[a3]
                            
                            # Was it bad?
                            was_bad = bool(ap_bad[idx])
                            
                            # What is the new color at p?
                            # p is one of a0, a1, a2, a3.
                            # We replace the color at p with c.
                            if a0 == p:
                                nc0, nc1, nc2, nc3 = c, c1, c2, c3
                            elif a1 == p:
                                nc0, nc1, nc2, nc3 = c0, c, c2, c3
                            elif a2 == p:
                                nc0, nc1, nc2, nc3 = c0, c1, c, c3
                            else: # a3 == p
                                nc0, nc1, nc2, nc3 = c0, c1, c2, c
                            
                            # Is it bad now?
                            is_bad_now = bool(bad_lookup[nc0, nc1, nc2, nc3])
                            
                            if was_bad and not is_bad_now:
                                delta -= 1
                            elif not was_bad and is_bad_now:
                                delta += 1
                            
                            if delta > best_delta:
                                # Pruning: if delta is already worse than best, stop
                                # But we need to be careful, we want min delta.
                                # If delta >= best_delta, we can break?
                                # Yes, because further APs can only add more delta (positive or negative).
                                # Wait, if we have fixed some, delta is negative. If we break new ones, delta is positive.
                                # If current partial sum is already >= best_delta, it might still decrease?
                                # No, the "fixed" contributions are negative. The "broken" are positive.
                                # It's possible to have a net negative.
                                # So we can't simply prune on partial sum unless we know the max possible negative contribution.
                                # Given small size, just finish the loop.
                                pass
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_mv = (p, c)
                
                if best_mv is None:
                    # Should not happen unless k=1 or something weird
                    break
                    
                p, c = best_mv
                # Apply move
                colors[p] = c
                
                # Update ap_bad and vset for affected APs
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
                    if stall > 100:
                        # Kick: random move
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

    # search
    result = None
    budgets = {3: 25, 4: 15, 5: 8, 6: 5, 7: 4, 8: 3}
    for k in range(3, 9):
        if time.time() > DEADLINE - 3:
            break
        b = budgets.get(k, 5)
        rem = DEADLINE - 3 - time.time()
        b = min(b, rem)
        if b <= 0.5:
            break
        res = try_k(k, b)
        if res is not None:
            ok = True
            for ap in aps:
                if len({res[ap[0]], res[ap[1]], res[ap[2]], res[ap[3]]}) < 3:
                    ok = False
                    break
            if ok:
                result = res
                break
    if result is not None:
        # Final polish: try to reduce distinct colors further
        polish_end = t0 + DEADLINE
        for _ in range(50):
            if time.time() > polish_end:
                break
            du = len(set(result))
            # Try to eliminate one color by remapping
            colors_used = sorted(set(result))
            for ci in range(len(colors_used)):
                target = colors_used[ci]
                # Try to remap all positions with color target to another color
                other_colors = [c for c in colors_used if c != target]
                if not other_colors:
                    break
                for nc in other_colors:
                    trial = result[:]
                    for i in range(n):
                        if trial[i] == target:
                            trial[i] = nc
                    ok = True
                    for ap in aps:
                        if len({trial[ap[0]], trial[ap[1]], trial[ap[2]], trial[ap[3]]}) < 3:
                            ok = False
                            break
                    if ok:
                        result = trial
                        break
                if len(set(result)) < du:
                    break
    if result is not None:
        return result
    # fallback greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
