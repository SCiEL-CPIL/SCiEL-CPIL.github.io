"""
Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows corresponding to targets 1..128.
Each row: list of 1..30 distinct positive 2^a*3^b*5^c terms summing to target.
No chosen term divides another. Minimize total summands.
"""
import time

_START = time.time()
DEADLINE = 85  # leave 5s margin before 90s kill


def construct():
    t0 = time.time()

    # Generate all smooth numbers up to 128
    smooth_set = set()
    for a in range(0, 8):
        v2 = 1 << a
        for b in range(0, 6):
            v23 = v2 * (3 ** b)
            if v23 > 128:
                break
            for c in range(0, 5):
                v = v23 * (5 ** c)
                if v > 128:
                    break
                if v >= 1:
                    smooth_set.add(v)
    smooth = sorted(smooth_set)
    n_smooth = len(smooth)
    smooth_idx = {s: i for i, s in enumerate(smooth)}

    # Precompute smooth lookup
    is_sm = [False] * 129
    for s in smooth:
        is_sm[s] = True

    # Precompute pair antichain validity
    pair_ok = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                pair_ok[i][j] = False
                pair_ok[j][i] = False

    # Precompute compatible masks (bitmask of indices that are compatible with i)
    compatible_mask = [0] * n_smooth
    for i in range(n_smooth):
        mask = 0
        for j in range(n_smooth):
            if i != j and pair_ok[i][j]:
                mask |= (1 << j)
        compatible_mask[i] = mask

    # Precompute min_k_sum and max_k_sum tables for fast lookup
    # min_k_sum[idx][k] = min sum of k distinct smooth numbers with index >= idx
    # max_k_sum[idx][k] = max sum of k distinct smooth numbers with index >= idx
    max_k = 30
    min_k_sum = [[0] * (max_k + 1) for _ in range(n_smooth + 1)]
    max_k_sum = [[0] * (max_k + 1) for _ in range(n_smooth + 1)]
    
    for idx in range(n_smooth - 1, -1, -1):
        for k in range(1, max_k + 1):
            if k == 1:
                min_k_sum[idx][k] = smooth[idx]
                max_k_sum[idx][k] = smooth[idx]
            else:
                # Option 1: don't use smooth[idx], use k from idx+1
                # Option 2: use smooth[idx] + (k-1) from idx+1
                if idx + 1 <= n_smooth:
                    min_k_sum[idx][k] = min_k_sum[idx + 1][k] if k <= max_k else float('inf')
                    max_k_sum[idx][k] = max_k_sum[idx + 1][k] if k <= max_k else 0
                    if k - 1 >= 0 and idx + 1 <= n_smooth:
                        min_k_sum[idx][k] = min(min_k_sum[idx][k], smooth[idx] + min_k_sum[idx + 1][k - 1])
                        max_k_sum[idx][k] = max(max_k_sum[idx][k], smooth[idx] + max_k_sum[idx + 1][k - 1])
                else:
                    min_k_sum[idx][k] = float('inf')
                    max_k_sum[idx][k] = 0

    # Memoized backtracking search with better state representation
    # State: (remaining, min_idx, frozenset_of_chosen_values) -> result or None
    memo = {}

    def _search(rem, min_idx, chosen_set, depth, max_depth):
        """Find max_depth - depth more smooth numbers (indices >= min_idx)
        that sum to rem and are antichain-compatible with all chosen."""
        if time.time() > DEADLINE:
            return None

        if depth == max_depth:
            return sorted(chosen_set) if rem == 0 else None
        if rem <= 0:
            return None

        # Pruning: min possible sum with remaining terms
        need = max_depth - depth
        if min_idx <= n_smooth:
            min_possible = min_k_sum[min_idx][need]
            if min_possible > rem:
                return None
            max_possible = max_k_sum[min_idx][need]
            if max_possible < rem:
                return None

        # Cache key - use frozenset of chosen values for better memoization
        cache_key = (rem, min_idx, frozenset(chosen_set), depth)
        if cache_key in memo:
            return memo[cache_key]

        result = None
        for idx in range(min_idx, n_smooth):
            s = smooth[idx]
            if s > rem:
                break

            # Check antichain with all chosen using bitmask
            ok = True
            for c in chosen_set:
                if s % c == 0 or c % s == 0:
                    ok = False
                    break
            if not ok:
                continue

            chosen_set.add(s)
            sub = _search(rem - s, idx + 1, chosen_set, depth + 1, max_depth)
            if sub is not None:
                result = sub
                break
            chosen_set.discard(s)

        memo[cache_key] = result
        return result

    best_terms = [None] * 129
    best_count = [999] * 129

    # For each target, find minimal representation
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE - 2:
            break

        # Try 1-term
        if is_sm[n]:
            best_terms[n] = [n]
            best_count[n] = 1
            continue

        # Try 2-term
        found = False
        for i in range(n_smooth):
            s1 = smooth[i]
            if s1 >= n:
                break
            s2 = n - s1
            if s2 <= s1:
                continue
            if not is_sm[s2]:
                continue
            j = smooth_idx.get(s2, -1)
            if j >= 0 and pair_ok[i][j]:
                best_terms[n] = [s1, s2]
                best_count[n] = 2
                found = True
                break
        if found:
            continue

        # Try 3-term
        found = False
        for i in range(n_smooth):
            s1 = smooth[i]
            if s1 * 3 >= n:
                break
            rem1 = n - s1
            for j in range(i + 1, n_smooth):
                s2 = smooth[j]
                if s2 * 2 >= rem1:
                    break
                s3 = rem1 - s2
                if s3 <= s2:
                    break
                if not is_sm[s3]:
                    continue
                k = smooth_idx.get(s3, -1)
                if k >= 0 and pair_ok[i][j] and pair_ok[i][k] and pair_ok[j][k]:
                    best_terms[n] = [s1, s2, s3]
                    best_count[n] = 3
                    found = True
                    break
            if found:
                break
        if found:
            continue

        # Try 4-term using backtracking (faster than O(n^4) brute force)
        found = False
        result = _search(n, 0, set(), 0, 4)
        if result is not None:
            best_terms[n] = result
            best_count[n] = 4
            found = True
        if found:
            continue

        # Try 5+ terms with memoized backtracking
        found = False
        for max_terms in range(5, 31):
            if time.time() - t0 > DEADLINE - 2:
                break
            result = _search(n, 0, set(), 0, max_terms)
            if result is not None:
                best_terms[n] = result
                best_count[n] = max_terms
                found = True
                break

        if not found:
            # Last resort fallback
            if is_sm[n]:
                best_terms[n] = [n]
                best_count[n] = 1
            else:
                # Try any valid pair
                for i in range(n_smooth):
                    s1 = smooth[i]
                    if s1 >= n:
                        break
                    s2 = n - s1
                    if s2 <= s1:
                        continue
                    if not is_sm[s2]:
                        continue
                    j = smooth_idx.get(s2, -1)
                    if j >= 0 and pair_ok[i][j]:
                        best_terms[n] = [s1, s2]
                        best_count[n] = 2
                        found = True
                        break
                if not found:
                    best_terms[n] = [1, n - 1] if n > 1 else [1]
                    best_count[n] = 2 if n > 1 else 1

    # Fill any missing targets
    for n in range(1, 129):
        if best_terms[n] is None:
            if is_sm[n]:
                best_terms[n] = [n]
                best_count[n] = 1
            else:
                for i in range(n_smooth):
                    s1 = smooth[i]
                    if s1 >= n:
                        break
                    s2 = n - s1
                    if s2 <= s1:
                        continue
                    if not is_sm[s2]:
                        continue
                    j = smooth_idx.get(s2, -1)
                    if j >= 0 and pair_ok[i][j]:
                        best_terms[n] = [s1, s2]
                        best_count[n] = 2
                        break
                if best_terms[n] is None:
                    best_terms[n] = [1, n - 1] if n > 1 else [1]
                    best_count[n] = 2 if n > 1 else 1

    return best_terms[1:129]
