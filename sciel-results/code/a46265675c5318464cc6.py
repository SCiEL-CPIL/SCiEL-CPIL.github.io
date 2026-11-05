import time
import random
import sys

_DEADLINE = 85.0
_MAX_SIZE = 48
_N = 512


def _check_violation(C, idx):
    """
    Check if C[idx] is the mean of any subset of >=2 OTHER elements.
    Uses bitset DP with early exit.
    """
    a = C[idx]
    n = len(C)
    if n < 3:
        return False

    diffs = [C[j] - a for j in range(n) if j != idx]
    m = len(diffs)
    if m < 2:
        return False

    # Quick check: size 2
    for i in range(m):
        for j in range(i + 1, m):
            if diffs[i] + diffs[j] == 0:
                return True

    # Quick check: size 3
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                if diffs[i] + diffs[j] + diffs[k] == 0:
                    return True

    # Calculate bounds
    max_pos = sum(d for d in diffs if d > 0)
    max_neg = -sum(d for d in diffs if d < 0)
    offset = max_neg
    total_span = max_pos + max_neg + 1

    # If span is too large, fall back to small subset checks only
    if total_span > 100000:
        # Check size 4 with meet-in-the-middle
        if m >= 4:
            pair_sums = {}
            for i in range(m):
                for j in range(i + 1, m):
                    s = diffs[i] + diffs[j]
                    if s not in pair_sums:
                        pair_sums[s] = []
                    pair_sums[s].append((i, j))
            for i in range(m):
                for j in range(i + 1, m):
                    target = -diffs[i] - diffs[j]
                    if target in pair_sums:
                        for (k, l) in pair_sums[target]:
                            if len({i, j, k, l}) == 4:
                                return True
        return False

    target_bit = 1 << offset
    mask = (1 << total_span) - 1

    dp = [0] * (m + 1)
    dp[0] = target_bit

    for d in diffs:
        for k in range(m, 0, -1):
            if dp[k - 1] == 0:
                continue
            if d >= 0:
                dp[k] |= (dp[k - 1] << d) & mask
            else:
                dp[k] |= (dp[k - 1] >> (-d)) & mask
            # Early exit for any subset size >= 2
            if k >= 2 and (dp[k] & target_bit):
                return True

    return False


def _is_valid(C):
    for i in range(len(C)):
        if _check_violation(C, i):
            return False
    return True


def _fast_incremental_check(C, new_elem):
    """
    Fast check for whether adding new_elem to C creates a violation.
    Uses O(1) set lookups instead of O(n^2) pair-sum dictionaries.
    """
    a = new_elem
    n = len(C)
    if n < 2:
        return True
    C_set = set(C)

    # 1. new_elem is mean of 2 others: C[i] + C[j] = 2*a
    # For each C[i], check if 2*a - C[i] is in C_set and != C[i]
    for i in range(n):
        partner = 2 * a - C[i]
        if partner in C_set and partner != C[i]:
            return False

    # 2. new_elem is mean of 3 others: C[i] + C[j] + C[k] = 3*a
    # For each pair (i,j), check if 3*a - C[i] - C[j] is in C_set and distinct
    for i in range(n):
        for j in range(i + 1, n):
            rem = 3 * a - C[i] - C[j]
            if rem in C_set and rem != C[i] and rem != C[j]:
                return False

    # 3. new_elem is mean of 4 others: C[i] + C[j] + C[k] + C[l] = 4*a
    # For each pair (i,j), need C[k]+C[l] = 4*a - C[i] - C[j]
    # Check via pair-sum existence: for target T, check if any pair sums to T
    # Use a set of pair sums for O(1) lookup
    if n >= 4:
        pair_sum_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                pair_sum_set.add(C[i] + C[j])
        for i in range(n):
            for j in range(i + 1, n):
                target = 4 * a - C[i] - C[j]
                if target in pair_sum_set:
                    # Need to verify 4 distinct elements exist
                    # Check if there's a pair (k,l) with k,l > j or different from i,j
                    # Simple check: if target can be formed by a pair not involving i or j
                    # For speed, just accept (false positives will be caught by _is_valid)
                    return False

    # 4. C[i] is mean of {new_elem, y}: 2*C[i] = a + y => y = 2*C[i] - a
    for i in range(n):
        y = 2 * C[i] - a
        if y in C_set and y != C[i]:
            return False

    # 5. C[i] is mean of {new_elem, y, z}: 3*C[i] = a + y + z => y + z = 3*C[i] - a
    # For each i, need two distinct elements y,z from C (excluding C[i]) with y+z = 3*C[i]-a
    for i in range(n):
        target_sum = 3 * C[i] - a
        for j in range(n):
            if C[j] == C[i]:
                continue
            z = target_sum - C[j]
            if z in C_set and z != C[i] and z != C[j]:
                return False

    # 6. C[i] is mean of {new_elem, y, z, w}: 4*C[i] = a + y + z + w
    # Need 3 distinct elements y,z,w from C (excluding C[i]) with y+z+w = 4*C[i]-a
    # For each i, check all pairs (y,z) from C\{C[i]}, see if w = 4*C[i]-a-y-z is in C\{C[i],C[j],z}
    if n >= 3:
        for i in range(n):
            target_sum3 = 4 * C[i] - a
            # Collect elements excluding C[i]
            others = [C[j] for j in range(n) if C[j] != C[i]]
            no = len(others)
            if no < 3:
                continue
            for p in range(no):
                for q in range(p + 1, no):
                    w = target_sum3 - others[p] - others[q]
                    if w in C_set and w != C[i] and w != others[p] and w != others[q]:
                        return False

    return True


def _greedy_construct(order, time_limit):
    selected = []
    start = time.time()

    for x in order:
        if time.time() - start > time_limit:
            break
        if len(selected) >= 48:
            break

        if _fast_incremental_check(selected, x):
            C = selected + [x]
            if _is_valid(C):
                selected.append(x)

    return selected


def _local_search_improve(C, time_limit):
    """
    Try to improve C by various moves with multiple passes:
    - Add elements
    - Remove 1, add 2 (net +1)
    - Remove 2, add 3 (net +1)
    """
    start = time.time()
    C = C[:]

    # Multiple passes: keep trying to add until no improvement
    for _pass in range(3):
        if time.time() - start > time_limit:
            break
        if len(C) >= _MAX_SIZE:
            break

        # Phase 1: Try to add elements (full scan)
        improved = False
        for x in range(1, _N + 1):
            if time.time() - start > time_limit:
                break
            if x in C:
                continue
            if len(C) >= _MAX_SIZE:
                break
            if _fast_incremental_check(C, x):
                C_new = C + [x]
                if _is_valid(C_new):
                    C = C_new
                    improved = True
        if not improved:
            break

        # Phase 2: Remove 1, add 2 (net +1)
        if len(C) < _MAX_SIZE:
            improved2 = False
            for i in range(len(C)):
                if time.time() - start > time_limit:
                    break
                C_without_i = C[:i] + C[i + 1:]
                for x in range(1, _N + 1):
                    if time.time() - start > time_limit:
                        break
                    if x in C_without_i:
                        continue
                    C_temp = C_without_i + [x]
                    if not _fast_incremental_check(C_without_i, x):
                        continue
                    for y in range(x + 1, _N + 1):
                        if time.time() - start > time_limit:
                            break
                        if y in C_temp:
                            continue
                        C_candidate = C_temp + [y]
                        if _fast_incremental_check(C_temp, y):
                            if _is_valid(C_candidate):
                                C = C_candidate
                                improved2 = True
                                break
                    if improved2:
                        break
                if improved2:
                    break
            if not improved2:
                break

        # Phase 3: Remove 2, add 3 (net +1)
        if len(C) < _MAX_SIZE and time.time() - start < time_limit - 1:
            improved3 = False
            for i in range(len(C)):
                if time.time() - start > time_limit:
                    break
                for j in range(i + 1, len(C)):
                    if time.time() - start > time_limit:
                        break
                    C_without = C[:i] + C[i + 1:j] + C[j + 1:]
                    for x in range(1, _N + 1):
                        if time.time() - start > time_limit:
                            break
                        if x in C_without:
                            continue
                        C_temp1 = C_without + [x]
                        if not _fast_incremental_check(C_without, x):
                            continue
                        for y in range(x + 1, _N + 1):
                            if time.time() - start > time_limit:
                                break
                            if y in C_temp1:
                                continue
                            C_temp2 = C_temp1 + [y]
                            if not _fast_incremental_check(C_temp1, y):
                                continue
                            for z in range(y + 1, _N + 1):
                                if time.time() - start > time_limit:
                                    break
                                if z in C_temp2:
                                    continue
                                C_candidate = C_temp2 + [z]
                                if _fast_incremental_check(C_temp2, z):
                                    if _is_valid(C_candidate):
                                        C = C_candidate
                                        improved3 = True
                                        break
                                if improved3:
                                    break
                            if improved3:
                                break
                        if improved3:
                            break
                    if improved3:
                        break
                if improved3:
                    break
            if not improved3:
                break

    return C


def _make_structured_orders():
    """Generate diverse structured orders."""
    orders = []

    # 1. Lo-Hi alternating
    lo, hi = 1, 512
    oh = []
    while lo <= hi:
        oh.append(lo)
        if lo != hi:
            oh.append(hi)
        lo += 1
        hi -= 1
    orders.append(oh)

    # 2. Hi-Lo alternating
    lo, hi = 1, 512
    oh = []
    while lo <= hi:
        oh.append(hi)
        if lo != hi:
            oh.append(lo)
        lo += 1
        hi -= 1
    orders.append(oh)

    # 3. Multiplicative pattern: 2^i * 3^j
    mult = []
    for i in range(10):
        for j in range(7):
            v = (1 << i) * (3 ** j)
            if 1 <= v <= 512:
                mult.append(v)
    mult = sorted(set(mult))
    rest = [x for x in range(1, 513) if x not in set(mult)]
    random.Random(789).shuffle(rest)
    orders.append(mult + rest)

    # 4. Power-of-2 first
    pow2 = [1 << i for i in range(10) if (1 << i) <= 512]
    rest = [x for x in range(1, 513) if x not in set(pow2)]
    random.Random(789).shuffle(rest)
    orders.append(pow2 + rest)

    # 5. Spread order
    base = list(range(1, 513, 10))
    random.Random(123).shuffle(base)
    remaining = [x for x in range(1, 513) if x not in set(base)]
    random.Random(456).shuffle(remaining)
    orders.append(base + remaining)

    # 6. Cluster order
    rng = random.Random(42)
    cluster_centers = list(range(10, 520, 30))
    rng.shuffle(cluster_centers)
    cluster_order = []
    for c in cluster_centers:
        spread = rng.randint(5, 15)
        for d in range(-spread, spread + 1):
            x = c + d
            if 1 <= x <= 512:
                cluster_order.append(x)
    seen = set()
    unique_cluster = []
    for x in cluster_order:
        if x not in seen:
            seen.add(x)
            unique_cluster.append(x)
    remaining = [x for x in range(1, 513) if x not in seen]
    rng.shuffle(remaining)
    orders.append(unique_cluster + remaining)

    # 7. Increasing
    orders.append(list(range(1, 513)))

    # 8. Decreasing
    orders.append(list(range(512, 0, -1)))

    return orders


def construct():
    start_time = time.time()
    best_set = []
    best_size = 0

    # Generate candidate orders
    structured_orders = _make_structured_orders()

    # Random orders
    random_orders = []
    for seed in range(20):
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        random_orders.append(order)

    all_orders = structured_orders + random_orders

    # Run greedy on each order
    for i, order in enumerate(all_orders):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 15:
            break

        time_limit = min(2.0, _DEADLINE - elapsed - 12)
        if time_limit < 0.5:
            break

        result = _greedy_construct(order, time_limit)

        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]

        if best_size == _MAX_SIZE:
            break

    # Local search improvement (multiple passes)
    if best_size < _MAX_SIZE and time.time() - start_time < _DEADLINE - 5:
        remaining_time = _DEADLINE - time.time() - 3
        if remaining_time > 2:
            improved = _local_search_improve(best_set, remaining_time)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved

    # Try more random orders if time permits
    if best_size < _MAX_SIZE and time.time() - start_time < _DEADLINE - 8:
        for seed in range(50, 150):
            if time.time() - start_time > _DEADLINE - 5:
                break
            rng = random.Random(seed)
            order = list(range(1, 513))
            rng.shuffle(order)
            result = _greedy_construct(order, 1.0)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result[:]
                # Quick local search on new best
                if best_size < _MAX_SIZE and time.time() - start_time < _DEADLINE - 5:
                    rem = _DEADLINE - time.time() - 3
                    if rem > 1:
                        improved = _local_search_improve(best_set, rem)
                        if len(improved) > best_size:
                            best_size = len(improved)
                            best_set = improved
                if best_size == _MAX_SIZE:
                    break

    if not best_set:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]

    return best_set
