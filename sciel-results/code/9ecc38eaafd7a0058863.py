def construct():
    import time
    import random
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

    # For each position, which APs does it appear in, and at which index
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for i, p in enumerate(ap):
            pos_aps[p].append((idx, i))

    def count_distinct(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        if c0 == c1 == c2 == c3:
            return 1
        if c0 == c1 == c2:
            return 2 if c3 != c0 else 1
        if c0 == c1 == c3:
            return 2 if c2 != c0 else 1
        if c0 == c2 == c3:
            return 2 if c1 != c0 else 1
        if c1 == c2 == c3:
            return 2 if c0 != c1 else 1
        if c0 == c1:
            return 3 if c2 != c0 and c3 != c0 and c2 != c3 else 2
        if c0 == c2:
            return 3 if c1 != c0 and c3 != c0 and c1 != c3 else 2
        if c0 == c3:
            return 3 if c1 != c0 and c2 != c0 and c1 != c2 else 2
        if c1 == c2:
            return 3 if c0 != c1 and c3 != c1 and c0 != c3 else 2
        if c1 == c3:
            return 3 if c0 != c1 and c2 != c1 and c0 != c2 else 2
        if c2 == c3:
            return 3 if c0 != c2 and c1 != c2 and c0 != c1 else 2
        return 4

    def try_k(k, budget):
        end = t0 + budget
        best = None
        best_du = 10**9
        rng = random.Random(42)

        while time.time() < end:
            # Random initialization
            colors = [rng.randrange(k) for _ in range(n)]

            # Precompute distinct counts
            dist = [0] * m
            vset = set()
            for i in range(m):
                d = count_distinct(colors, aps[i])
                dist[i] = d
                if d < 3:
                    vset.add(i)

            stall = 0
            while time.time() < end and vset:
                # Pick a violating AP
                ei = rng.choice(list(vset))
                ap = aps[ei]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                cur_set = {c0, c1, c2, c3}

                # Candidate moves: change a position in this AP to a color not in cur_set
                cand = []
                for p in ap:
                    for c in range(k):
                        if c not in cur_set:
                            cand.append((p, c))

                if not cand:
                    # No color outside cur_set available; try random
                    p = rng.choice(ap)
                    c = rng.randrange(k)
                    cand = [(p, c)]

                # Evaluate best move
                best_delta = 10**9
                best_mv = None
                for (p, c) in cand:
                    dlt = 0
                    old_c = colors[p]
                    for (idx, pi) in pos_aps[p]:
                        a2 = aps[idx]
                        was = idx in vset
                        # Compute new distinct count
                        new_vals = list(colors[a2])
                        new_vals[pi] = c
                        nd = count_distinct(new_vals, (0, 1, 2, 3)) if False else None
                        # Just compute directly
                        vals = []
                        for j in range(4):
                            if a2[j] == p:
                                vals.append(c)
                            else:
                                vals.append(colors[a2[j]])
                        # Count distinct
                        d0 = vals[0]
                        d1 = vals[1]
                        d2 = vals[2]
                        d3 = vals[3]
                        if d0 == d1 == d2 == d3:
                            nd = 1
                        elif d0 == d1 == d2:
                            nd = 2 if d3 != d0 else 1
                        elif d0 == d1 == d3:
                            nd = 2 if d2 != d0 else 1
                        elif d0 == d2 == d3:
                            nd = 2 if d1 != d0 else 1
                        elif d1 == d2 == d3:
                            nd = 2 if d0 != d1 else 1
                        elif d0 == d1:
                            nd = 3 if (d2 != d0 and d3 != d0 and d2 != d3) else 2
                        elif d0 == d2:
                            nd = 3 if (d1 != d0 and d3 != d0 and d1 != d3) else 2
                        elif d0 == d3:
                            nd = 3 if (d1 != d0 and d2 != d0 and d1 != d2) else 2
                        elif d1 == d2:
                            nd = 3 if (d0 != d1 and d3 != d1 and d0 != d3) else 2
                        elif d1 == d3:
                            nd = 3 if (d0 != d1 and d2 != d1 and d0 != d2) else 2
                        elif d2 == d3:
                            nd = 3 if (d0 != d2 and d1 != d2 and d0 != d1) else 2
                        else:
                            nd = 4
                        now = nd < 3
                        if was and not now:
                            dlt -= 1
                        elif (not was) and now:
                            dlt += 1
                    if dlt < best_delta:
                        best_delta = dlt
                        best_mv = (p, c)

                if best_mv is None:
                    break

                p, c = best_mv
                old = colors[p]
                colors[p] = c
                for (idx, pi) in pos_aps[p]:
                    a2 = aps[idx]
                    vals = []
                    for j in range(4):
                        if a2[j] == p:
                            vals.append(c)
                        else:
                            vals.append(colors[a2[j]])
                    d0, d1, d2, d3 = vals
                    if d0 == d1 == d2 == d3:
                        nd = 1
                    elif d0 == d1 == d2:
                        nd = 2 if d3 != d0 else 1
                    elif d0 == d1 == d3:
                        nd = 2 if d2 != d0 else 1
                    elif d0 == d2 == d3:
                        nd = 2 if d1 != d0 else 1
                    elif d1 == d2 == d3:
                        nd = 2 if d0 != d1 else 1
                    elif d0 == d1:
                        nd = 3 if (d2 != d0 and d3 != d0 and d2 != d3) else 2
                    elif d0 == d2:
                        nd = 3 if (d1 != d0 and d3 != d0 and d1 != d3) else 2
                    elif d0 == d3:
                        nd = 3 if (d1 != d0 and d2 != d0 and d1 != d2) else 2
                    elif d1 == d2:
                        nd = 3 if (d0 != d1 and d3 != d1 and d0 != d3) else 2
                    elif d1 == d3:
                        nd = 3 if (d0 != d1 and d2 != d1 and d0 != d2) else 2
                    elif d2 == d3:
                        nd = 3 if (d0 != d2 and d1 != d2 and d0 != d1) else 2
                    else:
                        nd = 4
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
                if stall > 2000:
                    # Random kick
                    for _ in range(5):
                        p2 = rng.randrange(n)
                        c2 = rng.randrange(k)
                        if c2 == colors[p2]:
                            continue
                        colors[p2] = c2
                        for (idx, pi) in pos_aps[p2]:
                            a2 = aps[idx]
                            vals = []
                            for j in range(4):
                                if a2[j] == p2:
                                    vals.append(c2)
                                else:
                                    vals.append(colors[a2[j]])
                            d0, d1, d2, d3 = vals
                            if d0 == d1 == d2 == d3:
                                nd = 1
                            elif d0 == d1 == d2:
                                nd = 2 if d3 != d0 else 1
                            elif d0 == d1 == d3:
                                nd = 2 if d2 != d0 else 1
                            elif d0 == d2 == d3:
                                nd = 2 if d1 != d0 else 1
                            elif d1 == d2 == d3:
                                nd = 2 if d0 != d1 else 1
                            elif d0 == d1:
                                nd = 3 if (d2 != d0 and d3 != d0 and d2 != d3) else 2
                            elif d0 == d2:
                                nd = 3 if (d1 != d0 and d3 != d0 and d1 != d3) else 2
                            elif d0 == d3:
                                nd = 3 if (d1 != d0 and d2 != d0 and d1 != d2) else 2
                            elif d1 == d2:
                                nd = 3 if (d0 != d1 and d3 != d1 and d0 != d3) else 2
                            elif d1 == d3:
                                nd = 3 if (d0 != d1 and d2 != d1 and d0 != d2) else 2
                            elif d2 == d3:
                                nd = 3 if (d0 != d2 and d1 != d2 and d0 != d1) else 2
                            else:
                                nd = 4
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
                    return best
        return best

    # Try 3 colors first
    result = try_k(3, 70)
    if result is not None:
        # Verify
        for ap in aps:
            if len({result[ap[0]], result[ap[1]], result[ap[2]], result[ap[3]]}) < 3:
                break
        else:
            return result

    # Fallback to 4 colors
    result = try_k(4, 5)
    if result is not None:
        for ap in aps:
            if len({result[ap[0]], result[ap[1]], result[ap[2]], result[ap[3]]}) < 3:
                break
        else:
            return result

    # Greedy fallback
    colors = []
    for x in range(1, 129):
        placed = False
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                prev = [x - 1 - j * d for j in range(1, 4)]
                if all(p >= 1 for p in prev):
                    prev_colors = [colors[p - 1] for p in prev]
                    if len(set(prev_colors + [color])) < 3:
                        ok = False
                        break
            if ok:
                colors.append(color)
                placed = True
                break
        if not placed:
            colors.append(0)
    return colors
