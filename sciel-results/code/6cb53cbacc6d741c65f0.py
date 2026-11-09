def construct():
    import time, random
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
    # local refs
    aps_l = aps
    pos_l = pos_aps
    m_l = m
    n_l = n

    def count_distinct_4(colors, ap):
        """Count distinct colors in a 4-term AP."""
        x0 = colors[ap[0]]; x1 = colors[ap[1]]
        x2 = colors[ap[2]]; x3 = colors[ap[3]]
        # count distinct without set allocation
        cnt = 1
        if x1 != x0: cnt += 1
        if x2 != x0 and x2 != x1: cnt += 1
        if x3 != x0 and x3 != x1 and x3 != x2: cnt += 1
        return cnt

    def try_k(k, budget):
        end = t0 + budget
        best = None
        best_du = 10**9
        rng = random
        while time.time() < end:
            # init
            if rng.random() < 0.5:
                colors = [i % k for i in range(n_l)]
            else:
                colors = [rng.randrange(k) for _ in range(n_l)]
            # Precompute distinct counts for all APs
            dist = [0] * m_l
            vset = set()
            for i in range(m_l):
                d = count_distinct_4(colors, aps_l[i])
                dist[i] = d
                if d < 3:
                    vset.add(i)
            stall = 0
            steps = 0
            while time.time() < end and vset:
                steps += 1
                # pick a violating edge
                ei = rng.choice(list(vset))
                ap = aps_l[ei]
                # Get current colors in this AP
                c0 = colors[ap[0]]; c1 = colors[ap[1]]
                c2 = colors[ap[2]]; c3 = colors[ap[3]]
                # Build set of colors present
                cur_set = set()
                cur_set.add(c0)
                cur_set.add(c1)
                cur_set.add(c2)
                cur_set.add(c3)
                # candidate moves: vertex in ap, color not in cur_set
                cand = []
                for p in ap:
                    for c in range(k):
                        if c not in cur_set:
                            cand.append((p, c))
                if not cand:
                    cand = [(ap[0], rng.randrange(k))]
                # evaluate best delta using incremental count
                best_delta = 10**9
                best_mv = None
                for (p, c) in cand:
                    dlt = 0
                    old_c = colors[p]
                    for idx in pos_l[p]:
                        a2 = aps_l[idx]
                        was = idx in vset
                        # Compute new distinct count efficiently
                        # We know the old count is dist[idx]
                        # Changing position p from old_c to c
                        # Need to check if old_c or c are unique in this AP
                        x0 = colors[a2[0]]; x1 = colors[a2[1]]
                        x2 = colors[a2[2]]; x3 = colors[a2[3]]
                        # Count occurrences of old_c and c in the AP
                        # For efficiency, just recompute the 4 values
                        vals = [x0, x1, x2, x3]
                        # Replace position p
                        pi = a2.index(p) if p in a2 else -1
                        # Actually, let's just recompute directly
                        new_vals = []
                        for q in a2:
                            if q == p:
                                new_vals.append(c)
                            else:
                                new_vals.append(colors[q])
                        # Count distinct in new_vals
                        nc = 1
                        for j in range(1, 4):
                            if new_vals[j] != new_vals[0]:
                                if new_vals[j] != new_vals[1] if j > 1 else True:
                                    if j == 1 or new_vals[j] != new_vals[1]:
                                        if j == 2 or new_vals[j] != new_vals[2]:
                                            if j == 3 or new_vals[j] != new_vals[3]:
                                                pass
                        # Simpler: use a small set
                        ns = {new_vals[0], new_vals[1], new_vals[2], new_vals[3]}
                        nd = len(ns)
                        now = nd < 3
                        if was and not now:
                            dlt -= 1
                        elif (not was) and now:
                            dlt += 1
                    if dlt < best_delta:
                        best_delta = dlt
                        best_mv = (p, c)
                p, c = best_mv
                old = colors[p]
                colors[p] = c
                for idx in pos_l[p]:
                    a2 = aps_l[idx]
                    nd = count_distinct_4(colors, a2)
                    dist[idx] = nd
                    now = nd < 3
                    was = idx in vset
                    if was and not now:
                        vset.discard(idx)
                    elif (not was) and now:
                        vset.add(idx)
                if best_delta < 0:
                    stall = 0
                else:
                    stall += 1
                if stall > 1500:
                    # kick
                    for _ in range(10):
                        p2 = rng.randrange(n_l)
                        c2 = rng.randrange(k)
                        if c2 == colors[p2]:
                            continue
                        colors[p2] = c2
                        for idx in pos_l[p2]:
                            a2 = aps_l[idx]
                            nd = count_distinct_4(colors, a2)
                            dist[idx] = nd
                            now = nd < 3
                            was = idx in vset
                            if was and not now:
                                vset.discard(idx)
                            elif (not was) and now:
                                vset.add(idx)
                    stall = 0
            if not vset:
                du = len(set(colors))
                if du < best_du:
                    best_du = du
                    best = colors[:]
                if best_du == k and k == 3:
                    # can't do better than 3 (2 impossible)
                    return best
        return best

    # search
    result = None
    budgets = {3: 16, 4: 12, 5: 8, 6: 6, 7: 5, 8: 4}
    for k in range(3, 9):
        if time.time() > DEADLINE - 3:
            break
        b = budgets.get(k, 5)
        # cap to remaining
        rem = DEADLINE - 3 - time.time()
        b = min(b, rem)
        if b <= 0.5:
            break
        res = try_k(k, b)
        if res is not None:
            # validate
            ok = True
            for ap in aps:
                if len({res[ap[0]], res[ap[1]], res[ap[2]], res[ap[3]]}) < 3:
                    ok = False
                    break
            if ok:
                result = res
                break
    if result is not None:
        # Try to reduce number of distinct colors via local search
        # If we have more than 3 colors, try to merge some
        used = set(result)
        if len(used) > 3:
            # Try to map colors to fewer distinct values
            # For each pair of colors, try merging
            color_list = sorted(used)
            improved = True
            while improved and time.time() < DEADLINE - 2:
                improved = False
                for i in range(len(color_list)):
                    for j in range(i+1, len(color_list)):
                        ci = color_list[i]
                        cj = color_list[j]
                        # Try mapping cj -> ci
                        trial = [ci if x == cj else x for x in result]
                        ok = True
                        for ap in aps:
                            if len({trial[ap[0]], trial[ap[1]], trial[ap[2]], trial[ap[3]]}) < 3:
                                ok = False
                                break
                        if ok:
                            result = trial
                            # Remove cj from color_list
                            color_list.remove(cj)
                            improved = True
                            break
                    if improved:
                        break
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
