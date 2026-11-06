import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Step 1: Generate all smooth numbers (2^a * 3^b * 5^c) <= 128
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        for b in range(0, 6):
            for c in range(0, 5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= v <= 128 and v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    smooth_idx = {v: i for i, v in enumerate(smooth)}

    # Step 2: Precompute divisibility conflicts as bitmask
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(n_smooth):
            if i != j and smooth[j] % si == 0:
                conflict_mask[i] |= (1 << j)

    # best_for_sum[s] = (count, tuple_of_indices)
    best_for_sum = [None] * 129

    # Size 1
    for i in range(n_smooth):
        s = smooth[i]
        if s <= 128:
            best_for_sum[s] = (1, (i,))

    # Size 2
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s = si + sj
            if s > 128:
                break
            if conflict_mask[i] & (1 << j):
                continue
            if best_for_sum[s] is None or best_for_sum[s][0] > 2:
                best_for_sum[s] = (2, (i, j))

    # Size 3
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s2 = si + sj
            if s2 > 128:
                break
            if conflict_mask[i] & (1 << j):
                continue
            for k in range(j + 1, n_smooth):
                sk = smooth[k]
                s3 = s2 + sk
                if s3 > 128:
                    break
                if (conflict_mask[i] & (1 << k)) or (conflict_mask[j] & (1 << k)):
                    continue
                if best_for_sum[s3] is None or best_for_sum[s3][0] > 3:
                    best_for_sum[s3] = (3, (i, j, k))

    # Size 4
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s2 = si + sj
            if s2 > 128:
                break
            if conflict_mask[i] & (1 << j):
                continue
            for k in range(j + 1, n_smooth):
                sk = smooth[k]
                s3 = s2 + sk
                if s3 > 128:
                    break
                if (conflict_mask[i] & (1 << k)) or (conflict_mask[j] & (1 << k)):
                    continue
                for l in range(k + 1, n_smooth):
                    sl = smooth[l]
                    s4 = s3 + sl
                    if s4 > 128:
                        break
                    if ((conflict_mask[i] & (1 << l)) or (conflict_mask[j] & (1 << l)) or (conflict_mask[k] & (1 << l))):
                        continue
                    if best_for_sum[s4] is None or best_for_sum[s4][0] > 4:
                        best_for_sum[s4] = (4, (i, j, k, l))

    # Size 5
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s2 = si + sj
            if s2 > 128:
                break
            if conflict_mask[i] & (1 << j):
                continue
            for k in range(j + 1, n_smooth):
                sk = smooth[k]
                s3 = s2 + sk
                if s3 > 128:
                    break
                if (conflict_mask[i] & (1 << k)) or (conflict_mask[j] & (1 << k)):
                    continue
                for l in range(k + 1, n_smooth):
                    sl = smooth[l]
                    s4 = s3 + sl
                    if s4 > 128:
                        break
                    if ((conflict_mask[i] & (1 << l)) or (conflict_mask[j] & (1 << l)) or (conflict_mask[k] & (1 << l))):
                        continue
                    for m in range(l + 1, n_smooth):
                        sm = smooth[m]
                        s5 = s4 + sm
                        if s5 > 128:
                            break
                        if ((conflict_mask[i] & (1 << m)) or (conflict_mask[j] & (1 << m)) or (conflict_mask[k] & (1 << m)) or (conflict_mask[l] & (1 << m))):
                            continue
                        if best_for_sum[s5] is None or best_for_sum[s5][0] > 5:
                            best_for_sum[s5] = (5, (i, j, k, l, m))

    # Check if all targets 1..128 are covered
    uncovered = [n for n in range(1, 129) if best_for_sum[n] is None]

    if uncovered:
        # Targeted DFS for uncovered targets, trying sizes 6+
        for target in uncovered:
            if time.time() - t0 > DEADLINE:
                break

            best = None
            for max_size in range(6, 15):
                if time.time() - t0 > DEADLINE:
                    break

                found = None
                selected = []

                def dfs(start_idx, remaining, size_left):
                    nonlocal found
                    if found is not None:
                        return
                    if size_left == 0:
                        if remaining == 0:
                            found = list(selected)
                        return
                    if remaining <= 0:
                        return
                    if n_smooth - start_idx < size_left:
                        return

                    # Pruning: minimum possible sum
                    min_sum = 0
                    for k in range(size_left):
                        if start_idx + k >= n_smooth:
                            return
                        min_sum += smooth[start_idx + k]
                    if min_sum > remaining:
                        return

                    for idx in range(start_idx, n_smooth):
                        val = smooth[idx]
                        if val > remaining:
                            break
                        # Check conflicts
                        ok = True
                        for s_idx in selected:
                            if conflict_mask[s_idx] & (1 << idx):
                                ok = False
                                break
                        if not ok:
                            continue
                        selected.append(idx)
                        dfs(idx + 1, remaining - val, size_left - 1)
                        selected.pop()
                        if found is not None:
                            return

                dfs(0, target, max_size)
                if found is not None:
                    best = tuple(found)
                    break

            if best is not None:
                best_for_sum[target] = (len(best), best)

    # Step 4: Build answers
    answers = []
    for n in range(1, 129):
        if best_for_sum[n] is not None:
            count, indices = best_for_sum[n]
            row = [smooth[i] for i in indices]
            row.sort()
            answers.append(row)
        else:
            # Fallback
            if n in smooth_set:
                answers.append([n])
            else:
                answers.append([1, n - 1])

    return answers
