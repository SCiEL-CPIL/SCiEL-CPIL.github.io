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
    # Precompute arrays for vectorized ops
    ap_arr = np.array(aps, dtype=np.int32)  # shape (m, 4)
    pos_l = pos_aps
    m_l = m
    n_l = n
    aps_l = aps

    def count_distinct4(colors, a, b, c, d):
        x0 = colors[a]; x1 = colors[b]; x2 = colors[c]; x3 = colors[d]
        if x0 == x1:
            if x0 == x2:
                if x0 == x3: return 1
                return 2
            if x0 == x3: return 2
            if x2 == x3: return 2
            return 3
        else:
            if x2 == x0:
                if x3 == x0: return 2
                if x3 == x1: return 3
                return 3
            elif x2 == x1:
                if x3 == x1: return 2
                if x3 == x0: return 3
                return 3
            else:
                if x3 == x0: return 3
                if x3 == x1: return 3
                if x3 == x2: return 3
                return 4

    def is_bad(colors, ap):
        return count_distinct4(colors, ap[0], ap[1], ap[2], ap[3]) < 3

    def compute_vset(colors):
        vset = set()
        for i in range(m_l):
            ap = aps_l[i]
            if is_bad(colors, ap):
                vset.add(i)
        return vset

    def try_k(k, budget):
        end = t0 + budget
        best = None
        best_du = 10**9
        rng = random
        # Precompute pos arrays as numpy for vectorized delta
        pos_arr = [np.array(pos_l[p], dtype=np.int32) for p in range(n_l)]
        
        while time.time() < end:
            # init
            if rng.random() < 0.5:
                colors_np = np.array([i % k for i in range(n_l)], dtype=np.int32)
            else:
                colors_np = np.array([rng.randrange(k) for _ in range(n_l)], dtype=np.int32)
            vset = set()
            for i in range(m_l):
                ap = aps_l[i]
                if len({colors_np[ap[0]], colors_np[ap[1]], colors_np[ap[2]], colors_np[ap[3]]}) < 3:
                    vset.add(i)
            stall = 0
            while time.time() < end and vset:
                # pick a violating edge
                ei = rng.choice(list(vset))
                ap = aps_l[ei]
                cur = {int(colors_np[ap[0]]), int(colors_np[ap[1]]), int(colors_np[ap[2]]), int(colors_np[ap[3]])}
                # candidate moves: vertex in ap, color not in cur
                cand = []
                for p in ap:
                    for c in range(k):
                        if c not in cur:
                            cand.append((p, c))
                if not cand:
                    cand = [(ap[0], rng.randrange(k))]
                # evaluate best delta using vectorized ops
                best_delta = 10**9
                best_mv = None
                for (p, c) in cand:
                    pos_idx = pos_arr[p]
                    if len(pos_idx) == 0:
                        dlt = 0
                    else:
                        # Get all APs containing p
                        sub_aps = ap_arr[pos_idx]  # shape (num_aps, 4)
                        # Current colors of all positions in these APs
                        cols = colors_np[sub_aps]  # shape (num_aps, 4)
                        # Current distinct count (vectorized)
                        # For each row, count distinct values
                        # A row is bad if it has < 3 distinct values
                        # Compute: for each row, is it currently bad?
                        # distinct count = 4 - (number of equal pairs that collapse)
                        # Simpler: use unique per row via sort
                        sorted_cols = np.sort(cols, axis=1)
                        # bad if all 4 same, or 3+2, or 2+2
                        # all same: sorted_cols[:,0]==sorted_cols[:,3]
                        # 3 same + 1 diff: (sorted_cols[:,0]==sorted_cols[:,2]) | (sorted_cols[:,1]==sorted_cols[:,3])
                        # 2+2: sorted_cols[:,0]==sorted_cols[:,1] and sorted_cols[:,2]==sorted_cols[:,3]
                        all_same = (sorted_cols[:,0] == sorted_cols[:,3])
                        three_same = ((sorted_cols[:,0] == sorted_cols[:,2]) & (sorted_cols[:,0] != sorted_cols[:,3])) | \
                                     ((sorted_cols[:,1] == sorted_cols[:,3]) & (sorted_cols[:,0] != sorted_cols[:,1]))
                        two_two = (sorted_cols[:,0] == sorted_cols[:,1]) & (sorted_cols[:,2] == sorted_cols[:,3]) & \
                                  (sorted_cols[:,0] != sorted_cols[:,1])
                        currently_bad = all_same | three_same | two_two
                        # After change: replace color at position p in each AP
                        # Find which column in sub_aps corresponds to p
                        col_mask = (sub_aps == p)  # shape (num_aps, 4)
                        new_cols = cols.copy()
                        new_cols[col_mask] = c
                        sorted_new = np.sort(new_cols, axis=1)
                        all_same_new = (sorted_new[:,0] == sorted_new[:,3])
                        three_same_new = ((sorted_new[:,0] == sorted_new[:,2]) & (sorted_new[:,0] != sorted_new[:,3])) | \
                                         ((sorted_new[:,1] == sorted_new[:,3]) & (sorted_new[:,0] != sorted_new[:,1]))
                        two_two_new = (sorted_new[:,0] == sorted_new[:,1]) & (sorted_new[:,2] == sorted_new[:,3]) & \
                                      (sorted_new[:,0] != sorted_new[:,1])
                        now_bad = all_same_new | three_same_new | two_two_new
                        # Delta: -1 for fixed (was bad, not now), +1 for broken (was ok, now bad)
                        fixed = currently_bad & ~now_bad
                        broken = ~currently_bad & now_bad
                        dlt = int(broken.sum() - fixed.sum())
                    if dlt < best_delta:
                        best_delta = dlt
                        best_mv = (p, c)
                p, c = best_mv
                colors_np[p] = c
                # Update vset for affected APs
                for idx in pos_l[p]:
                    a2 = aps_l[idx]
                    now = len({int(colors_np[a2[0]]), int(colors_np[a2[1]]), int(colors_np[a2[2]]), int(colors_np[a2[3]])}) < 3
                    was = idx in vset
                    if was and not now:
                        vset.discard(idx)
                    elif (not was) and now:
                        vset.add(idx)
                if best_delta < 0:
                    stall = 0
                else:
                    stall += 1
                if stall > 400:
                    # kick
                    for _ in range(6):
                        p2 = rng.randrange(n_l)
                        c2 = rng.randrange(k)
                        if c2 == colors_np[p2]:
                            continue
                        colors_np[p2] = c2
                        for idx in pos_l[p2]:
                            a2 = aps_l[idx]
                            now = len({int(colors_np[a2[0]]), int(colors_np[a2[1]]), int(colors_np[a2[2]]), int(colors_np[a2[3]])}) < 3
                            was = idx in vset
                            if was and not now:
                                vset.discard(idx)
                            elif (not was) and now:
                                vset.add(idx)
                    stall = 0
            if not vset:
                du = len(set(int(x) for x in colors_np))
                if du < best_du:
                    best_du = du
                    best = [int(x) for x in colors_np]
                if best_du == k and k == 3:
                    return best
        return best

    # search
    result = None
    budgets = {3: 20, 4: 16, 5: 10, 6: 7, 7: 5, 8: 4}
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
