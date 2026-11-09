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

    def bad_at(colors, ap):
        a,b,c,d = ap
        x0=colors[a]; x1=colors[b]; x2=colors[c]; x3=colors[d]
        # count distinct
        s = {x0,x1,x2,x3}
        return len(s) < 3

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
            vset = set(i for i in range(m_l) if bad_at(colors, aps_l[i]))
            stall = 0
            steps = 0
            while time.time() < end and vset:
                steps += 1
                # pick a violating edge
                ei = rng.choice(list(vset))
                ap = aps_l[ei]
                cur = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                # candidate moves: vertex in ap, color not in cur
                cand = []
                for p in ap:
                    for c in range(k):
                        if c not in cur:
                            cand.append((p, c))
                if not cand:
                    # all colors present? can't be, since len(cur)<3 and k>=3 -> there is outside
                    # but if k==len(cur) impossible since len(cur)<=2<k for k>=3
                    cand = [(ap[0], rng.randrange(k))]
                # evaluate best delta
                best_delta = 10**9
                best_mv = None
                for (p, c) in cand:
                    dlt = 0
                    for idx in pos_l[p]:
                        a2 = aps_l[idx]
                        was = idx in vset
                        # after
                        s = set()
                        for q in a2:
                            s.add(c if q == p else colors[q])
                        now = len(s) < 3
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
                    now = bad_at(colors, a2)
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
                            now = bad_at(colors, a2)
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
