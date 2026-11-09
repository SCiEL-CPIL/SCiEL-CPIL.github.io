import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 512
    MAXC = 48

    def build_dp(S):
        """
        Builds DP bitsets for subset sums of S.
        dp[k] is a bitset where bit s is 1 if s is the sum of exactly k elements from S.
        """
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1  # Sum 0 with 0 elements
        for a in S:
            # Update in reverse to avoid using the same element multiple times in one step
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def verify(S):
        """
        Verifies the entire set S.
        No element in S is the mean of any subset of at least 2 OTHER elements.
        """
        S = list(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i + 1:]
            m = len(sub)
            if m < 2:
                continue
            dp = build_dp(sub)
            maxs = sum(sub)
            for k in range(2, m + 1):
                t = k * a
                if t > maxs:
                    break
                if (dp[k] >> t) & 1:
                    return False
        return True

    # Precompute 3-AP involvement degrees
    # deg[x] = number of 3-term arithmetic progressions involving x
    deg = [0] * (N + 1)
    for a in range(1, N - 1):
        # c must have same parity as a
        for c in range(a + 2, N + 1, 2):
            b = (a + c) // 2
            if 1 <= b <= N:
                deg[a] += 1
                deg[b] += 1
                deg[c] += 1

    # Also compute a "pair conflict" metric: for each x, count how many pairs (u,v)
    # in [1,N] \ {x} have mean x, i.e., u + v = 2x.
    # This is the most direct constraint on whether x can be included.
    pair_conflict = [0] * (N + 1)
    for x in range(1, N + 1):
        # Count pairs (u, v) with u < v, u,v != x, u + v = 2x
        # u ranges from 1 to x-1 (since u < v and u+v=2x means u < x < v)
        # v = 2x - u
        for u in range(1, x):
            v = 2 * x - u
            if v > x and v <= N:
                pair_conflict[x] += 1

    best = []
    best_sz = 0

    def try_greedy(order):
        """Greedily build a set following the given order, using DP-based verification."""
        cur = []
        cache = {}

        def get_dps(S):
            key = tuple(sorted(S))
            r = cache.get(key)
            if r is not None:
                return r

            dp_full = build_dp(S)
            maxfull = sum(S)
            dp_without = {}
            n = len(S)

            for i in range(n):
                sub = S[:i] + S[i + 1:]
                dp_without[S[i]] = (build_dp(sub), maxfull - S[i])

            r = (dp_full, maxfull, dp_without)
            cache[key] = r
            return r

        def test_add(S, x, dp_full, maxfull, dp_without):
            """
            Checks if x can be added to S without violating the mean condition.
            1. x is not the mean of any subset of S (size >= 2).
            2. No element a in S becomes the mean of any subset of S U {x} containing x.
            """
            n = len(S)

            # Check 1: x not the mean of any subset of S (size>=2)
            for k in range(2, n + 1):
                t = k * x
                if t > maxfull:
                    break
                if (dp_full[k] >> t) & 1:
                    return False

            # Check 2: no a in S becomes mean of a subset containing x
            for a in S:
                dpo, maxo = dp_without[a]
                m = n - 1  # Size of S \ {a}
                # T contains x. T' = T \ {x} is subset of S \ {a} of size j.
                # Sum(T') = (j+1)*a - x.
                for j in range(1, m + 1):
                    t = (j + 1) * a - x
                    if t < 0 or t > maxo:
                        continue
                    if (dpo[j] >> t) & 1:
                        return False
            return True

        for x in order:
            if len(cur) >= MAXC:
                break
            if time.time() > deadline - 5:
                break
            dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)

        return cur

    def local_search(S):
        """Try to improve S by adding elements or swapping."""
        if time.time() > deadline - 5:
            return S

        S = list(S)
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]

        # Sort unused by pair_conflict ascending (safer elements first)
        unused.sort(key=lambda x: pair_conflict[x])

        cache = {}

        def get_dps(S):
            key = tuple(sorted(S))
            r = cache.get(key)
            if r is not None:
                return r

            dp_full = build_dp(S)
            maxfull = sum(S)
            dp_without = {}
            n = len(S)

            for i in range(n):
                sub = S[:i] + S[i + 1:]
                dp_without[S[i]] = (build_dp(sub), maxfull - S[i])

            r = (dp_full, maxfull, dp_without)
            cache[key] = r
            return r

        def test_add(S, x, dp_full, maxfull, dp_without):
            n = len(S)

            # Check 1: x not the mean of any subset of S (size>=2)
            for k in range(2, n + 1):
                t = k * x
                if t > maxfull:
                    break
                if (dp_full[k] >> t) & 1:
                    return False

            # Check 2: no a in S becomes mean of a subset containing x
            for a in S:
                dpo, maxo = dp_without[a]
                m = n - 1
                for j in range(1, m + 1):
                    t = (j + 1) * a - x
                    if t < 0 or t > maxo:
                        continue
                    if (dpo[j] >> t) & 1:
                        return False
            return True

        improved = True
        no_improve = 0

        while improved and time.time() < deadline - 3:
            improved = False
            dps = get_dps(S)

            # Try to add an unused element
            added = False
            for u in unused:
                if len(S) >= MAXC:
                    break
                if time.time() > deadline - 3:
                    break
                if test_add(S, u, dps[0], dps[1], dps[2]):
                    S.append(u)
                    Sset.add(u)
                    unused.remove(u)
                    improved = True
                    no_improve = 0
                    added = True
                    break

            if len(S) >= MAXC:
                break

            if not added:
                # Try swaps: remove s from S, add u from unused
                # Sort S by pair_conflict descending to try removing most constrained first
                S_sorted = sorted(S, key=lambda x: pair_conflict[x], reverse=True)
                swapped = False

                max_swaps_to_try = min(len(unused), 50)
                if time.time() > deadline - 8:
                    max_swaps_to_try = min(len(unused), 20)

                for u in unused[:max_swaps_to_try]:
                    if time.time() > deadline - 3:
                        break
                    for s in S_sorted:
                        idx = S.index(s)
                        newS = S[:idx] + S[idx + 1:]
                        dps2 = get_dps(newS)
                        if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
                            S = newS + [u]
                            Sset.remove(s)
                            Sset.add(u)
                            unused.remove(u)
                            unused.append(s)
                            unused.sort(key=lambda x: pair_conflict[x])
                            improved = True
                            swapped = True
                            no_improve = 0
                            break
                    if swapped:
                        break

                if not swapped:
                    no_improve += 1
                    if no_improve > 8:
                        break

        return S

    # Phase 1: Multiple greedy restarts with different orderings
    candidates = list(range(1, N + 1))

    # Strategy 1: Sort by pair_conflict ascending
    order1 = sorted(candidates, key=lambda x: (pair_conflict[x], deg[x], random.random()))

    # Strategy 2: Sort by deg ascending
    order2 = sorted(candidates, key=lambda x: (deg[x], pair_conflict[x], random.random()))

    num_restarts = 30

    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break

        if i % 3 == 0:
            order = order1[:]
        elif i % 3 == 1:
            order = order2[:]
        else:
            order = candidates[:]
            random.shuffle(order)

        # Try forward and reverse
        for rev in [False, True]:
            if time.time() > deadline - 20:
                break
            cur_order = order[::-1] if rev else order
            cur = try_greedy(cur_order)

            if len(cur) > best_sz:
                best_sz = len(cur)
                best = cur[:]

        if best_sz >= MAXC:
            break

    # Phase 2: Local search on the best found
    if best_sz < MAXC and time.time() < deadline - 10:
        S = local_search(best[:])
        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # Phase 3: If still not at max, try local search on other greedy results
    if best_sz < MAXC and time.time() < deadline - 5:
        # Generate a few more greedy solutions and local search them
        for i in range(5):
            if time.time() > deadline - 5:
                break
            order = candidates[:]
            random.shuffle(order)
            cur = try_greedy(order)
            if len(cur) >= best_sz - 2:  # Only local search if close to best
                S = local_search(cur)
                if verify(S) and len(S) > best_sz:
                    best_sz = len(S)
                    best = S[:]
            if best_sz >= MAXC:
                break

    # Final safety check
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
