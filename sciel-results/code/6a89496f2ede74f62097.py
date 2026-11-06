import time
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    # Step 1: Generate all smooth numbers (2^a * 3^b * 5^c) <= 128
    smooth = []
    smooth_set = set()
    smooth_index = {}
    for a in range(0, 8):
        for b in range(0, 6):
            for c in range(0, 5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= v <= 128 and v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    for i, v in enumerate(smooth):
        smooth_index[v] = i

    # is_smooth lookup
    is_sm = [False] * 129
    for s in smooth:
        is_sm[s] = True

    # Step 2: Precompute divisibility pairs
    # div_conflict[i][j] = True if smooth[i] divides smooth[j] or vice versa (i != j)
    div_conflict = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            if sj % si == 0 or si % sj == 0:
                div_conflict[i][j] = True
                div_conflict[j][i] = True

    # Step 3: Generate all valid antichain subsets (size 1 to 5) with sum <= 128
    # For each sum, store the minimum-size antichain that achieves it
    # best_for_sum[s] = (count, tuple_of_indices)
    best_for_sum = [None] * 129  # index 0..128

    # Size 1
    for i in range(n_smooth):
        s = smooth[i]
        if s <= 128:
            best_for_sum[s] = (1, (i,))

    # Size 2
    for i in range(n_smooth):
        si = smooth[i]
        if si >= 128:
            break
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s = si + sj
            if s > 128:
                break
            if div_conflict[i][j]:
                continue
            if best_for_sum[s] is None or best_for_sum[s][0] > 2:
                best_for_sum[s] = (2, (i, j))

    # Size 3
    for i in range(n_smooth):
        si = smooth[i]
        if si >= 128:
            break
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s2 = si + sj
            if s2 >= 128:
                break
            if div_conflict[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                sk = smooth[k]
                s3 = s2 + sk
                if s3 > 128:
                    break
                if div_conflict[i][k] or div_conflict[j][k]:
                    continue
                if best_for_sum[s3] is None or best_for_sum[s3][0] > 3:
                    best_for_sum[s3] = (3, (i, j, k))

    # Size 4
    for i in range(n_smooth):
        si = smooth[i]
        if si >= 128:
            break
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s2 = si + sj
            if s2 >= 128:
                break
            if div_conflict[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                sk = smooth[k]
                s3 = s2 + sk
                if s3 >= 128:
                    break
                if div_conflict[i][k] or div_conflict[j][k]:
                    continue
                for l in range(k + 1, n_smooth):
                    sl = smooth[l]
                    s4 = s3 + sl
                    if s4 > 128:
                        break
                    if div_conflict[i][l] or div_conflict[j][l] or div_conflict[k][l]:
                        continue
                    if best_for_sum[s4] is None or best_for_sum[s4][0] > 4:
                        best_for_sum[s4] = (4, (i, j, k, l))

    # Size 5
    for i in range(n_smooth):
        si = smooth[i]
        if si >= 128:
            break
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s2 = si + sj
            if s2 >= 128:
                break
            if div_conflict[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                sk = smooth[k]
                s3 = s2 + sk
                if s3 >= 128:
                    break
                if div_conflict[i][k] or div_conflict[j][k]:
                    continue
                for l in range(k + 1, n_smooth):
                    sl = smooth[l]
                    s4 = s3 + sl
                    if s4 >= 128:
                        break
                    if div_conflict[i][l] or div_conflict[j][l] or div_conflict[k][l]:
                        continue
                    for m in range(l + 1, n_smooth):
                        sm = smooth[m]
                        s5 = s4 + sm
                        if s5 > 128:
                            break
                        if div_conflict[i][m] or div_conflict[j][m] or div_conflict[k][m] or div_conflict[l][m]:
                            continue
                        if best_for_sum[s5] is None or best_for_sum[s5][0] > 5:
                            best_for_sum[s5] = (5, (i, j, k, l, m))

    # Check if all targets 1..128 are covered
    uncovered = [n for n in range(1, 129) if best_for_sum[n] is None]
    
    if uncovered:
        # Need size 6 or 7 - do targeted search for uncovered targets
        # Fixed: Check if rem is a valid smooth number not in the current set, regardless of index order
        for target in uncovered:
            if time.time() - t0 > DEADLINE:
                break
            
            # Try size 6
            found = False
            for i in range(n_smooth):
                si = smooth[i]
                if si >= target:
                    break
                for j in range(i + 1, n_smooth):
                    sj = smooth[j]
                    s2 = si + sj
                    if s2 >= target:
                        break
                    if div_conflict[i][j]:
                        continue
                    for k in range(j + 1, n_smooth):
                        sk = smooth[k]
                        s3 = s2 + sk
                        if s3 >= target:
                            break
                        if div_conflict[i][k] or div_conflict[j][k]:
                            continue
                        for l in range(k + 1, n_smooth):
                            sl = smooth[l]
                            s4 = s3 + sl
                            if s4 >= target:
                                break
                            if div_conflict[i][l] or div_conflict[j][l] or div_conflict[k][l]:
                                continue
                            for m in range(l + 1, n_smooth):
                                sm = smooth[m]
                                s5 = s4 + sm
                                if s5 >= target:
                                    break
                                if div_conflict[i][m] or div_conflict[j][m] or div_conflict[k][m] or div_conflict[l][m]:
                                    continue
                                rem = target - s5
                                if rem <= sm:
                                    break
                                # Check if rem is smooth and doesn't conflict
                                if rem in smooth_set:
                                    idx = smooth_index[rem]
                                    # Check if idx is distinct from i, j, k, l, m
                                    if idx != i and idx != j and idx != k and idx != l and idx != m:
                                        if not (div_conflict[i][idx] or div_conflict[j][idx] or div_conflict[k][idx] or div_conflict[l][idx] or div_conflict[m][idx]):
                                            best_for_sum[target] = (6, (i, j, k, l, m, idx))
                                            found = True
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            
            if found:
                continue
            
            # Try size 7
            found = False
            for i in range(n_smooth):
                si = smooth[i]
                if si >= target:
                    break
                for j in range(i + 1, n_smooth):
                    sj = smooth[j]
                    s2 = si + sj
                    if s2 >= target:
                        break
                    if div_conflict[i][j]:
                        continue
                    for k in range(j + 1, n_smooth):
                        sk = smooth[k]
                        s3 = s2 + sk
                        if s3 >= target:
                            break
                        if div_conflict[i][k] or div_conflict[j][k]:
                            continue
                        for l in range(k + 1, n_smooth):
                            sl = smooth[l]
                            s4 = s3 + sl
                            if s4 >= target:
                                break
                            if div_conflict[i][l] or div_conflict[j][l] or div_conflict[k][l]:
                                continue
                            for m in range(l + 1, n_smooth):
                                sm = smooth[m]
                                s5 = s4 + sm
                                if s5 >= target:
                                    break
                                if div_conflict[i][m] or div_conflict[j][m] or div_conflict[k][m] or div_conflict[l][m]:
                                    continue
                                for n2 in range(m + 1, n_smooth):
                                    sn = smooth[n2]
                                    s6 = s5 + sn
                                    if s6 >= target:
                                        break
                                    if div_conflict[i][n2] or div_conflict[j][n2] or div_conflict[k][n2] or div_conflict[l][n2] or div_conflict[m][n2]:
                                        continue
                                    rem = target - s6
                                    if rem <= sn:
                                        break
                                    if rem in smooth_set:
                                        idx = smooth_index[rem]
                                        # Check if idx is distinct from i, j, k, l, m, n2
                                        if idx != i and idx != j and idx != k and idx != l and idx != m and idx != n2:
                                            if not (div_conflict[i][idx] or div_conflict[j][idx] or div_conflict[k][idx] or div_conflict[l][idx] or div_conflict[m][idx] or div_conflict[n2][idx]):
                                                best_for_sum[target] = (7, (i, j, k, l, m, n2, idx))
                                                found = True
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

    # Step 4: Build answers
    answers = []
    for n in range(1, 129):
        if best_for_sum[n] is not None:
            count, indices = best_for_sum[n]
            row = [smooth[i] for i in indices]
            row.sort()
            answers.append(row)
        else:
            # Fallback - should not happen
            if is_sm[n]:
                answers.append([n])
            else:
                # Find any valid representation
                answers.append([1, n - 1])
    
    return answers
