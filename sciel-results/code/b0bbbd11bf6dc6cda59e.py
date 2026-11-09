def construct():
    import time, random
    import numpy as np
    t0 = time.time()
    DEADLINE = t0 + 50.0
    n = 128
    
    # Generate all nonconstant 4-term APs
    aps = []
    for a in range(1, n+1):
        max_d = (n - a) // 3
        for d in range(1, max_d+1):
            aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    m = len(aps)
    
    # Map position -> list of AP indices containing it
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    
    ap_arr = np.array(aps, dtype=np.int32)
    
    def count_distinct4(colors, a, b, c, d):
        vals = [colors[a], colors[b], colors[c], colors[d]]
        return len(set(vals))
    
    def is_bad(colors, ap):
        return count_distinct4(colors, ap[0], ap[1], ap[2], ap[3]) < 3
    
    def compute_vset(colors):
        vset = set()
        for i in range(m):
            if is_bad(colors, aps[i]):
                vset.add(i)
        return vset
    
    def verify(colors):
        for ap in aps:
            if len(set(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])) < 3:
                return False
        return True
    
    def try_k(k, budget):
        end = t0 + budget
        best = None
        best_du = 10**9
        rng = random.Random(42)
        
        # Precompute for vectorized delta computation
        pos_arr = [np.array(pos_aps[p], dtype=np.int32) for p in range(n)]
        
        def fast_bad_count(colors_np, ap_idx):
            """Count bad APs using vectorized ops"""
            sub = ap_arr[ap_idx]
            cols = colors_np[sub]
            sorted_cols = np.sort(cols, axis=1)
            all_same = (sorted_cols[:,0] == sorted_cols[:,3])
            three_same = ((sorted_cols[:,0] == sorted_cols[:,2]) & (sorted_cols[:,0] != sorted_cols[:,3])) | \
                         ((sorted_cols[:,1] == sorted_cols[:,3]) & (sorted_cols[:,0] != sorted_cols[:,1]))
            two_two = (sorted_cols[:,0] == sorted_cols[:,1]) & (sorted_cols[:,2] == sorted_cols[:,3]) & \
                      (sorted_cols[:,0] != sorted_cols[:,1])
            bad = all_same | three_same | two_two
            return int(bad.sum())
        
        def compute_delta(colors_np, p, c, pos_idx):
            """Compute change in bad count if we set position p to color c"""
            if len(pos_idx) == 0:
                return 0
            sub = ap_arr[pos_idx]
            cols = colors_np[sub]
            # Current bad count
            sorted_cols = np.sort(cols, axis=1)
            all_same = (sorted_cols[:,0] == sorted_cols[:,3])
            three_same = ((sorted_cols[:,0] == sorted_cols[:,2]) & (sorted_cols[:,0] != sorted_cols[:,3])) | \
                         ((sorted_cols[:,1] == sorted_cols[:,3]) & (sorted_cols[:,0] != sorted_cols[:,1]))
            two_two = (sorted_cols[:,0] == sorted_cols[:,1]) & (sorted_cols[:,2] == sorted_cols[:,3]) & \
                      (sorted_cols[:,0] != sorted_cols[:,1])
            cur_bad = all_same | three_same | two_two
            
            # New colors
            col_mask = (sub == p)
            new_cols = cols.copy()
            new_cols[col_mask] = c
            sorted_new = np.sort(new_cols, axis=1)
            all_same_new = (sorted_new[:,0] == sorted_new[:,3])
            three_same_new = ((sorted_new[:,0] == sorted_new[:,2]) & (sorted_new[:,0] != sorted_new[:,3])) | \
                             ((sorted_new[:,1] == sorted_new[:,3]) & (sorted_new[:,0] != sorted_new[:,1]))
            two_two_new = (sorted_new[:,0] == sorted_new[:,1]) & (sorted_new[:,2] == sorted_new[:,3]) & \
                          (sorted_new[:,0] != sorted_new[:,1])
            new_bad = all_same_new | three_same_new | two_two_new
            
            fixed = cur_bad & ~new_bad
            broken = ~cur_bad & new_bad
            return int(broken.sum() - fixed.sum())
        
        while time.time() < end:
            # Initialize with random or structured coloring
            if rng.random() < 0.3:
                colors_np = np.array([i % k for i in range(n)], dtype=np.int32)
            elif rng.random() < 0.5:
                colors_np = np.array([rng.randrange(k) for _ in range(n)], dtype=np.int32)
            else:
                # Try a structured pattern
                pattern = [0, 1, 2, 3, 1, 2, 0, 3] if k >= 4 else [0, 1, 2, 0, 1, 2, 1, 0]
                colors_np = np.array([pattern[i % len(pattern)] % k for i in range(n)], dtype=np.int32)
            
            vset = set()
            for i in range(m):
                ap = aps[i]
                if len({int(colors_np[ap[0]]), int(colors_np[ap[1]]), int(colors_np[ap[2]]), int(colors_np[ap[3]])}) < 3:
                    vset.add(i)
            
            stall = 0
            while time.time() < end and vset:
                # Pick a violating edge
                ei = rng.choice(list(vset))
                ap = aps[ei]
                cur_colors = [int(colors_np[ap[0]]), int(colors_np[ap[1]]), int(colors_np[ap[2]]), int(colors_np[ap[3]])]
                cur_set = set(cur_colors)
                
                # Candidate moves: change one position in AP to a color not in current set
                cand = []
                for pos in ap:
                    for c in range(k):
                        if c not in cur_set:
                            cand.append((pos, c))
                
                if not cand:
                    # Fallback: random change
                    pos = rng.choice(ap)
                    c = rng.randrange(k)
                    cand = [(pos, c)]
                
                # Evaluate best move
                best_delta = 10**9
                best_mv = None
                for (p, c) in cand:
                    pos_idx = pos_arr[p]
                    dlt = compute_delta(colors_np, p, c, pos_idx)
                    if dlt < best_delta:
                        best_delta = dlt
                        best_mv = (p, c)
                
                if best_mv is None:
                    break
                    
                p, c = best_mv
                old_c = int(colors_np[p])
                colors_np[p] = c
                
                # Update vset
                for idx in pos_aps[p]:
                    a2 = aps[idx]
                    now_bad = len({int(colors_np[a2[0]]), int(colors_np[a2[1]]), int(colors_np[a2[2]]), int(colors_np[a2[3]])}) < 3
                    was_bad = idx in vset
                    if was_bad and not now_bad:
                        vset.discard(idx)
                    elif (not was_bad) and now_bad:
                        vset.add(idx)
                
                if best_delta < 0:
                    stall = 0
                else:
                    stall += 1
                
                if stall > 300:
                    # Kick: random perturbations
                    for _ in range(8):
                        p2 = rng.randrange(n)
                        c2 = rng.randrange(k)
                        if c2 == int(colors_np[p2]):
                            continue
                        colors_np[p2] = c2
                        for idx in pos_aps[p2]:
                            a2 = aps[idx]
                            now_bad = len({int(colors_np[a2[0]]), int(colors_np[a2[1]]), int(colors_np[a2[2]]), int(colors_np[a2[3]])}) < 3
                            was_bad = idx in vset
                            if was_bad and not now_bad:
                                vset.discard(idx)
                            elif (not was_bad) and now_bad:
                                vset.add(idx)
                    stall = 0
            
            if not vset:
                du = len(set(int(x) for x in colors_np))
                if du < best_du:
                    best_du = du
                    best = [int(x) for x in colors_np]
                if best_du == k:
                    # Found valid with k colors
                    return best
        
        return best
    
    # Search from k=3 upward
    result = None
    budgets = {3: 18, 4: 14, 5: 10, 6: 6}
    for k in range(3, 7):
        if time.time() > DEADLINE - 5:
            break
        b = budgets.get(k, 5)
        rem = DEADLINE - 5 - time.time()
        b = min(b, rem)
        if b <= 1.0:
            break
        res = try_k(k, b)
        if res is not None:
            if verify(res):
                result = res
                break
    
    if result is not None:
        # Try to reduce colors further by remapping
        for _ in range(30):
            if time.time() > DEADLINE - 2:
                break
            du = len(set(result))
            colors_used = sorted(set(result))
            reduced = False
            for target in colors_used:
                others = [c for c in colors_used if c != target]
                if not others:
                    break
                for nc in others:
                    trial = result[:]
                    for i in range(n):
                        if trial[i] == target:
                            trial[i] = nc
                    if verify(trial):
                        result = trial
                        reduced = True
                        break
                if reduced:
                    break
            if not reduced:
                break
    
    if result is not None:
        return result
    
    # Fallback: greedy construction
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            ok = True
            # Check all 4-APs ending at x-1
            for d in range(1, (x-1)//3 + 1):
                a = x-1-3*d
                b = x-1-2*d
                c = x-1-d
                if a < 0 or b < 0 or c < 0:
                    continue
                ap_colors = [color, colors[a], colors[b], colors[c]]
                if len(set(ap_colors)) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
    
    return colors
