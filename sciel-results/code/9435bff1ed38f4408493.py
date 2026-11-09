def construct():
    import time, random
    t0 = time.time()
    DEADLINE = t0 + 50.0  # safety margin
    n = 128

    # Precompute all nonconstant 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            if a + 3 * d < n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    m = len(aps)

    # For each position, which APs does it participate in?
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)

    def count_distinct_fast(c0, c1, c2, c3):
        """Count distinct colors among 4 values without set allocation."""
        cnt = 1
        if c1 != c0:
            cnt += 1
        if c2 != c0 and c2 != c1:
            cnt += 1
        if c3 != c0 and c3 != c1 and c3 != c2:
            cnt += 1
        return cnt

    def try_k(k, budget, rng):
        """Try to find a valid k-coloring. Returns coloring or None."""
        end = t0 + budget
        best = None
        best_viol = m + 1

        while time.time() < end:
            # Random initialization
            if rng.random() < 0.5:
                colors = [i % k for i in range(n)]
            else:
                colors = [rng.randrange(k) for _ in range(n)]

            # Precompute violation status for all APs
            viol = [False] * m
            nviol = 0
            for i in range(m):
                ap = aps[i]
                d = count_distinct_fast(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
                if d < 3:
                    viol[i] = True
                    nviol += 1

            if nviol == 0:
                return colors

            stall = 0
            steps = 0
            while time.time() < end and nviol > 0:
                steps += 1
                if steps % 5000 == 0 and time.time() > end - 1:
                    break

                # Pick a random violating AP
                vi = rng.randrange(m)
                while not viol[vi]:
                    vi = rng.randrange(m)
                ap = aps[vi]

                # Current colors in this AP
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                cur_colors = set([c0, c1, c2, c3])

                # Candidate moves: change one position to a color not in cur_colors
                # This guarantees the AP gets at least 3 colors (since we add a new color)
                cand = []
                for p in ap:
                    for c in range(k):
                        if c not in cur_colors:
                            cand.append((p, c))

                if not cand:
                    # Fallback: random move
                    p = ap[rng.randrange(4)]
                    c = rng.randrange(k)
                    cand = [(p, c)]

                # Evaluate best move by delta in violation count
                best_delta = m + 1
                best_mv = None
                for (p, c) in cand:
                    old_c = colors[p]
                    if old_c == c:
                        continue
                    dlt = 0
                    for idx in pos_aps[p]:
                        a2 = aps[idx]
                        was_viol = viol[idx]
                        # Compute new distinct count
                        nc0 = colors[a2[0]] if a2[0] != p else c
                        nc1 = colors[a2[1]] if a2[1] != p else c
                        nc2 = colors[a2[2]] if a2[2] != p else c
                        nc3 = colors[a2[3]] if a2[3] != p else c
                        nd = count_distinct_fast(nc0, nc1, nc2, nc3)
                        now_viol = nd < 3
                        if was_viol and not now_viol:
                            dlt -= 1
                        elif (not was_viol) and now_viol:
                            dlt += 1
                    if dlt < best_delta:
                        best_delta = dlt
                        best_mv = (p, c)

                if best_mv is None:
                    # No improving move found, do random kick
                    for _ in range(rng.randint(1, 5)):
                        p = rng.randrange(n)
                        c = rng.randrange(k)
                        if c == colors[p]:
                            continue
                        colors[p] = c
                        for idx in pos_aps[p]:
                            a2 = aps[idx]
                            nc0 = colors[a2[0]] if a2[0] != p else c
                            nc1 = colors[a2[1]] if a2[1] != p else c
                            nc2 = colors[a2[2]] if a2[2] != p else c
                            nc3 = colors[a2[3]] if a2[3] != p else c
                            nd = count_distinct_fast(nc0, nc1, nc2, nc3)
                            now_viol = nd < 3
                            if viol[idx] and not now_viol:
                                viol[idx] = False
                                nviol -= 1
                            elif (not viol[idx]) and now_viol:
                                viol[idx] = True
                                nviol += 1
                    stall += 1
                    if stall > 200:
                        break
                    continue

                p, c = best_mv
                old = colors[p]
                colors[p] = c
                for idx in pos_aps[p]:
                    a2 = aps[idx]
                    nc0 = colors[a2[0]] if a2[0] != p else c
                    nc1 = colors[a2[1]] if a2[1] != p else c
                    nc2 = colors[a2[2]] if a2[2] != p else c
                    nc3 = colors[a2[3]] if a2[3] != p else c
                    nd = count_distinct_fast(nc0, nc1, nc2, nc3)
                    now_viol = nd < 3
                    if viol[idx] and not now_viol:
                        viol[idx] = False
                        nviol -= 1
                    elif (not viol[idx]) and now_viol:
                        viol[idx] = True
                        nviol += 1

                if best_delta < 0:
                    stall = 0
                else:
                    stall += 1

                if nviol == 0:
                    return colors

                if stall > 1000:
                    # Larger kick
                    for _ in range(10):
                        p2 = rng.randrange(n)
                        c2 = rng.randrange(k)
                        if c2 == colors[p2]:
                            continue
                        colors[p2] = c2
                        for idx in pos_aps[p2]:
                            a2 = aps[idx]
                            nc0 = colors[a2[0]] if a2[0] != p2 else c2
                            nc1 = colors[a2[1]] if a2[1] != p2 else c2
                            nc2 = colors[a2[2]] if a2[2] != p2 else c2
                            nc3 = colors[a2[3]] if a2[3] != p2 else c2
                            nd = count_distinct_fast(nc0, nc1, nc2, nc3)
                            now_viol = nd < 3
                            if viol[idx] and not now_viol:
                                viol[idx] = False
                                nviol -= 1
                            elif (not viol[idx]) and now_viol:
                                viol[idx] = True
                                nviol += 1
                    stall = 0

            # If we exited the inner loop without solving, check if valid
            if nviol == 0:
                return colors
            if nviol < best_viol:
                best_viol = nviol
                best = colors[:]

        return best

    def is_valid(colors):
        for ap in aps:
            if count_distinct_fast(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]) < 3:
                return False
        return True

    def reduce_colors(colors, max_colors, deadline):
        """Try to reduce the number of distinct colors in a valid coloring."""
        if len(set(colors)) <= max_colors:
            return colors

        used = sorted(set(colors))
        if len(used) <= max_colors:
            return colors

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(used)):
                for j in range(i + 1, len(used)):
                    ci, cj = used[i], used[j]
                    trial = [ci if x == cj else x for x in colors]
                    if is_valid(trial):
                        colors = trial
                        used = sorted(set(colors))
                        improved = True
                        break
                if improved:
                    break
        return colors

    # Main search: try increasing k
    rng = random.Random(42)
    result = None

    # Try 3 colors first with more time
    if time.time() < DEADLINE - 5:
        res = try_k(3, min(15, DEADLINE - time.time() - 3), rng)
        if res is not None and is_valid(res):
            result = res

    if result is None and time.time() < DEADLINE - 5:
        res = try_k(4, min(12, DEADLINE - time.time() - 3), rng)
        if res is not None and is_valid(res):
            result = res

    if result is None and time.time() < DEADLINE - 5:
        res = try_k(5, min(10, DEADLINE - time.time() - 3), rng)
        if res is not None and is_valid(res):
            result = res

    if result is None and time.time() < DEADLINE - 5:
        res = try_k(6, min(8, DEADLINE - time.time() - 3), rng)
        if res is not None and is_valid(res):
            result = res

    if result is None and time.time() < DEADLINE - 5:
        res = try_k(7, min(6, DEADLINE - time.time() - 3), rng)
        if res is not None and is_valid(res):
            result = res

    if result is None and time.time() < DEADLINE - 5:
        res = try_k(8, min(5, DEADLINE - time.time() - 3), rng)
        if res is not None and is_valid(res):
            result = res

    if result is not None:
        # Try to reduce colors
        result = reduce_colors(result, 3, DEADLINE - 2)
        return result

    # Fallback: greedy construction
    colors = []
    for x in range(n):
        # x is 0-indexed, position is x+1
        valid = True
        for color in range(128):
            ok = True
            # Check all 4-APs where this position is the last element
            for d in range(1, (x + 1) // 3 + 1):
                a = (x + 1) - 3 * d
                if a < 1:
                    continue
                # Positions (0-indexed): a-1, a+d-1, a+2d-1, x
                i1 = a - 1
                i2 = a + d - 1
                i3 = a + 2 * d - 1
                if i3 < 0 or i2 < 0 or i1 < 0:
                    continue
                c1 = colors[i1]
                c2 = colors[i2]
                c3 = colors[i3]
                if count_distinct_fast(c1, c2, c3, color) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                break
        else:
            # Should not happen, but safety
            colors.append(0)
    return colors
