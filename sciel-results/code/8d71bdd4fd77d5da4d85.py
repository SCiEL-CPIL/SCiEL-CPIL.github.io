# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict

def _is_valid_fast(arr, n):
    """Check no element is the mean of any subset (size>=2) of the other elements."""
    m = len(arr)
    if m < 3:
        return True
    pos = {v: i for i, v in enumerate(arr)}
    max_sum = m * n
    # dp[k] = bitset of sums achievable by k elements of the whole set
    dp = [0] * (m + 1)
    dp[0] = 1
    for v in arr:
        for k in range(m - 1, -1, -1):
            dp[k + 1] |= dp[k] << v
    # For each element, build the dp over all other elements by removing its contribution
    for idx, a in enumerate(arr):
        other = [arr[j] for j in range(m) if j != idx]
        d = [0] * m
        d[0] = 1
        for v in other:
            for k in range(len(other) - 1, -1, -1):
                d[k + 1] |= d[k] << v
        # a is the mean of a subset T of others iff (a*len(T) - sum(T)) is representable
        # equivalently sum(T) = a*len(T); use d directly:
        # a is mean of T iff sum(T) == a*len(T). But we also require that "a itself" is not in T.
        # Since T is over others, that's already satisfied.
        # Check: does d[k] have bit (a*k) set?  (subset sum exactly a*k of size k)
        for k in range(2, m):
            target = a * k
            if target > max_sum:
                break
            if (d[k] >> target) & 1:
                return False
    return True

def construct():
    deadline = time.time() + 50
    N = 512
    rng = random.Random(12345)
    best = None
    best_size = 0

    def try_greedy(order):
        nonlocal best, best_size
        chosen = []
        # maintain incremental validity with bitsets for the current chosen set
        # dp[k] = bitset of sums of k chosen elements
        dp = [0] * (49)
        dp[0] = 1
        for c in order:
            m = len(chosen)
            if m >= 48:
                break
            # test adding c
            # new dp:
            ndp = dp[:]  # copy up to m
            ok = True
            # compute new dp after adding c
            for k in range(m, -1, -1):
                ndp[k + 1] |= dp[k] << c
            # Now test each existing element a and the new element c
            # For new element c: subset of others (old chosen) summing to c*k for k>=2
            for k in range(2, m + 1):
                target = c * k
                if (ndp[k] >> target) & 1 and (target) and c not in []:
                    # but ndp includes c; need subsets of OTHERS = old chosen for c
                    # sum of k old elements == c*k
                    # dp[k] (old) has sums of k old elements
                    if (dp[k] >> target) & 1:
                        ok = False
                        break
            if ok:
                # for each existing a, subset of others (chosen without a plus c) summing to a*k, k>=2
                # We'll do it by checking: any subset of (chosen + c) not containing a, size>=2, sum = a*k
                # Use ndp[k] (sums of k from chosen+c) minus contributions that include a
                # subset of chosen+c not containing a of size k, sum s:
                # = (subset of chosen of size k sum s) OR (subset of chosen of size k-1 sum s-c, with c in subset)
                # So bitset B_k = dp[k] | (dp[k-1] << c)
                for a in chosen:
                    bad = False
                    for k in range(2, m + 1):
                        target = a * k
                        if target > 48 * N:
                            break
                        B_k = dp[k] | ((dp[k - 1] << c) if k - 1 >= 0 else 0)
                        if (B_k >> target) & 1:
                            bad = True
                            break
                    if bad:
                        ok = False
                        break
            if ok:
                chosen.append(c)
                dp = ndp
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best = chosen[:]
                if len(chosen) == 48:
                    return best
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = chosen[:]
        return best

    # Many random orders
    pool = list(range(1, N + 1))
    for it in range(4000):
        if time.time() > deadline:
            break
        rng.shuffle(pool)
        try_greedy(pool)
        if best_size >= 48:
            break
        # also try descending/ascending occasionally
    # local search: from best, try to improve by swap/insert
    if best is not None and best_size < 48:
        cur = best[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(pool)
            for c in pool:
                if c in cur:
                    continue
                # try replacing a random element
                # try inserting and removing one to keep same size, or just add if size<48
                # Simple: try to add c (size grows) via full check
                trial = cur + [c]
                if _is_valid_fast(trial, N):
                    cur = trial
                    improved = True
                    if len(cur) >= 48:
                        best = cur[:]
                        return best
                    break
        if len(cur) > best_size:
            best = cur[:]
            best_size = len(cur)

    if best is None:
        best = list(range(1, 49))
    return best
# EVOLVE-BLOCK-END
