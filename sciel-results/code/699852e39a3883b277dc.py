"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    max_target = 128

    # Generate all 2,3,5-smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        x = n
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(n)

    n_smooth = len(smooth)
    idx_of = {v: i for i, v in enumerate(smooth)}

    # Precompute compatibility bitmask: compat_mask[i] has bit j set if i and j are compatible
    compat_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] != 0 and smooth[i] % smooth[j] != 0:
                compat_mask[i] |= (1 << j)

    best_count = [max_target + 1] * (max_target + 1)
    best_terms = [None] * (max_target + 1)

    # Size 1
    for i, v in enumerate(smooth):
        if v <= max_target:
            best_count[v] = 1
            best_terms[v] = [v]

    # Size 2: all compatible pairs
    pair_list = []  # (sum, i, j)
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if (compat_mask[i] >> j) & 1:
                s = smooth[i] + smooth[j]
                if s <= max_target:
                    pair_list.append((s, i, j))
                    if 2 < best_count[s]:
                        best_count[s] = 2
                        best_terms[s] = [smooth[i], smooth[j]]

    # Size 3: for each compatible pair (i,j), try adding k > j
    triple_list = []
    for s, i, j in pair_list:
        # Find k > j compatible with both i and j
        for k in range(j + 1, n_smooth):
            if ((compat_mask[i] >> k) & 1) and ((compat_mask[j] >> k) & 1):
                s3 = s + smooth[k]
                if s3 <= max_target:
                    triple_list.append((s3, i, j, k))
                    if 3 < best_count[s3]:
                        best_count[s3] = 3
                        best_terms[s3] = [smooth[i], smooth[j], smooth[k]]

    # Check time
    if time.time() > deadline - 3.0:
        return _fill_remaining(smooth, best_count, best_terms, max_target, n_smooth, compat_mask, deadline)

    # Size 4: for each compatible triple (i,j,k), try adding l > k
    for s3, i, j, k in triple_list:
        for l in range(k + 1, n_smooth):
            if ((compat_mask[i] >> l) & 1) and ((compat_mask[j] >> l) & 1) and ((compat_mask[k] >> l) & 1):
                s4 = s3 + smooth[l]
                if s4 <= max_target:
                    if 4 < best_count[s4]:
                        best_count[s4] = 4
                        best_terms[s4] = [smooth[i], smooth[j], smooth[k], smooth[l]]

    # Check time
    if time.time() > deadline - 3.0:
        return _fill_remaining(smooth, best_count, best_terms, max_target, n_smooth, compat_mask, deadline)

    # For any remaining gaps, use targeted backtracking search
    gaps = [n for n in range(1, max_target + 1) if best_terms[n] is None]
    for target in gaps:
        if time.time() > deadline - 2.0:
            break
        found = _backtrack_search(smooth, target, n_smooth, compat_mask, min_terms=4, max_terms=8)
        if found:
            best_count[target] = len(found)
            best_terms[target] = found

    # Final fallback
    still_missing = [n for n in range(1, max_target + 1) if best_terms[n] is None]
    for target in still_missing:
        if time.time() > deadline - 1.0:
            break
        found = _backtrack_search(smooth, target, n_smooth, compat_mask, min_terms=4, max_terms=12)
        if found:
            best_count[target] = len(found)
            best_terms[target] = found
        else:
            # Last resort: try any representation
            for cnt in range(1, 15):
                cands = [x for x in smooth if x <= target]
                for terms in combinations(cands, cnt):
                    if sum(terms) == target:
                        # Verify antichain
                        ok = True
                        for a, b in combinations(terms, 2):
                            if b % a == 0 or a % b == 0:
                                ok = False
                                break
                        if ok:
                            best_count[target] = cnt
                            best_terms[target] = list(terms)
                            break
                if best_terms[target] is not None:
                    break
            if best_terms[target] is None:
                best_count[target] = 1
                best_terms[target] = [target] if target in idx_of else [1]

    return [best_terms[n] for n in range(1, max_target + 1)]


def _backtrack_search(smooth, target, n_smooth, compat_mask, min_terms, max_terms):
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
            # Check compatibility with all in current using bitmask
            compatible = True
            for j in current_indices:
                if not ((compat_mask[i] >> j) & 1):
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
        return [smooth[i] for i in best]
    return None


def _fill_remaining(smooth, best_count, best_terms, max_target, n_smooth, compat_mask, deadline):
    """Fill remaining targets using backtracking search."""
    for n in range(1, max_target + 1):
        if best_terms[n] is None:
            if time.time() > deadline - 1.0:
                break
            found = _backtrack_search(smooth, n, n_smooth, compat_mask, min_terms=1, max_terms=10)
            if found:
                best_count[n] = len(found)
                best_terms[n] = found
            else:
                # Last resort
                for cnt in range(1, 12):
                    cands = [x for x in smooth if x <= n]
                    for terms in combinations(cands, cnt):
                        if sum(terms) == n:
                            ok = True
                            for a, b in combinations(terms, 2):
                                if b % a == 0 or a % b == 0:
                                    ok = False
                                    break
                            if ok:
                                best_count[n] = cnt
                                best_terms[n] = list(terms)
                                break
                    if best_terms[n] is not None:
                        break
                if best_terms[n] is None:
                    best_count[n] = 1
                    best_terms[n] = [n] if n in smooth else [1]
    return [best_terms[n] for n in range(1, max_target + 1)]


# EVOLVE-BLOCK-END
