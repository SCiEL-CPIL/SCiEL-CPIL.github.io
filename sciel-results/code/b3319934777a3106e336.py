"""Combined approach: precompute all antichain subsets up to size 5, then targeted search for gaps."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 82.0

    max_target = 128

    # Generate all 2,3,5-smooth numbers <= 128
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        for b in range(0, 6):
            for c in range(0, 5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= v <= max_target and v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    idx_of = {v: i for i, v in enumerate(smooth)}

    # Precompute divisibility conflict matrix
    div_conflict = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            if sj % si == 0 or si % sj == 0:
                div_conflict[i][j] = True
                div_conflict[j][i] = True

    # best_for_sum[s] = (count, tuple_of_indices)
    best_for_sum = [None] * (max_target + 1)

    # Size 1
    for i in range(n_smooth):
        s = smooth[i]
        if s <= max_target:
            best_for_sum[s] = (1, (i,))

    # Size 2
    for i in range(n_smooth):
        si = smooth[i]
        if si >= max_target:
            break
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s = si + sj
            if s > max_target:
                break
            if div_conflict[i][j]:
                continue
            if best_for_sum[s] is None or best_for_sum[s][0] > 2:
                best_for_sum[s] = (2, (i, j))

    # Size 3
    for i in range(n_smooth):
        si = smooth[i]
        if si >= max_target:
            break
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s2 = si + sj
            if s2 >= max_target:
                break
            if div_conflict[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                sk = smooth[k]
                s3 = s2 + sk
                if s3 > max_target:
                    break
                if div_conflict[i][k] or div_conflict[j][k]:
                    continue
                if best_for_sum[s3] is None or best_for_sum[s3][0] > 3:
                    best_for_sum[s3] = (3, (i, j, k))

    # Size 4
    for i in range(n_smooth):
        si = smooth[i]
        if si >= max_target:
            break
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s2 = si + sj
            if s2 >= max_target:
                break
            if div_conflict[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                sk = smooth[k]
                s3 = s2 + sk
                if s3 >= max_target:
                    break
                if div_conflict[i][k] or div_conflict[j][k]:
                    continue
                for l in range(k + 1, n_smooth):
                    sl = smooth[l]
                    s4 = s3 + sl
                    if s4 > max_target:
                        break
                    if div_conflict[i][l] or div_conflict[j][l] or div_conflict[k][l]:
                        continue
                    if best_for_sum[s4] is None or best_for_sum[s4][0] > 4:
                        best_for_sum[s4] = (4, (i, j, k, l))

    # Size 5
    for i in range(n_smooth):
        si = smooth[i]
        if si >= max_target:
            break
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            s2 = si + sj
            if s2 >= max_target:
                break
            if div_conflict[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                sk = smooth[k]
                s3 = s2 + sk
                if s3 >= max_target:
                    break
                if div_conflict[i][k] or div_conflict[j][k]:
                    continue
                for l in range(k + 1, n_smooth):
                    sl = smooth[l]
                    s4 = s3 + sl
                    if s4 > max_target:
                        break
                    if div_conflict[i][l] or div_conflict[j][l] or div_conflict[k][l]:
                        continue
                    for m in range(l + 1, n_smooth):
                        sm = smooth[m]
                        s5 = s4 + sm
                        if s5 > max_target:
                            break
                        if div_conflict[i][m] or div_conflict[j][m] or div_conflict[k][m] or div_conflict[l][m]:
                            continue
                        if best_for_sum[s5] is None or best_for_sum[s5][0] > 5:
                            best_for_sum[s5] = (5, (i, j, k, l, m))

    # Check time
    if time.time() - t0 > DEADLINE - 5.0:
        return _build_result(best_for_sum, smooth, max_target)

    # For uncovered targets, do targeted backtracking search
    uncovered = [n for n in range(1, max_target + 1) if best_for_sum[n] is None]
    for target in uncovered:
        if time.time() - t0 > DEADLINE - 2.0:
            break
        found = _backtrack_search(smooth, target, n_smooth, div_conflict, min_terms=1, max_terms=10)
        if found:
            best_for_sum[target] = (len(found), tuple(found))

    # Final fallback for any still-missing targets
    still_missing = [n for n in range(1, max_target + 1) if best_for_sum[n] is None]
    for target in still_missing:
        if time.time() - t0 > DEADLINE - 1.0:
            break
        # Try brute force with small sizes
        cands = [i for i in range(n_smooth) if smooth[i] <= target]
        for cnt in range(1, 8):
            if time.time() - t0 > DEADLINE - 0.5:
                break
            for combo in combinations(cands, cnt):
                if sum(smooth[i] for i in combo) == target:
                    # Verify antichain
                    ok = True
                    for a in range(len(combo)):
                        for b in range(a + 1, len(combo)):
                            if div_conflict[combo[a]][combo[b]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        best_for_sum[target] = (cnt, combo)
                        break
            if best_for_sum[target] is not None:
                break
        if best_for_sum[target] is None:
            # Last resort: use target itself if smooth, else 1
            if target in smooth_set:
                best_for_sum[target] = (1, (idx_of[target],))
            else:
                best_for_sum[target] = (1, (0,))  # use smooth[0] = 1

    return _build_result(best_for_sum, smooth, max_target)


def _build_result(best_for_sum, smooth, max_target):
    """Build the 128-row result from best_for_sum."""
    result = []
    for n in range(1, max_target + 1):
        if best_for_sum[n] is not None:
            count, indices = best_for_sum[n]
            result.append([smooth[i] for i in indices])
        else:
            result.append([1])
    return result


def _backtrack_search(smooth, target, n_smooth, div_conflict, min_terms, max_terms):
    """Backtracking search for antichain representation of target."""
    best = None

    def backtrack(remaining, min_idx, current_indices):
        nonlocal best
        cur_len = len(current_indices)
        if cur_len > max_terms:
            return
        if remaining == 0:
            if cur_len >= min_terms:
                if best is None or cur_len < len(best):
                    best = list(current_indices)
            return
        if remaining < 0:
            return

        slots_left = max_terms - cur_len
        if slots_left <= 0:
            return
        if remaining < slots_left:
            return

        for i in range(min_idx, n_smooth):
            v = smooth[i]
            if v > remaining:
                break
            # Check compatibility with all in current
            compatible = True
            for j in current_indices:
                if div_conflict[i][j]:
                    compatible = False
                    break
            if not compatible:
                continue
            current_indices.append(i)
            backtrack(remaining - v, i + 1, current_indices)
            current_indices.pop()
            if best is not None and len(best) == min_terms:
                return

    backtrack(target, 0, [])
    if best:
        return best
    return None

# EVOLVE-BLOCK-END
