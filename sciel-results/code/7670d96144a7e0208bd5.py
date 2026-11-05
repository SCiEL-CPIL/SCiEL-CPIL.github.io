def construct():
    import time
    start_time = time.time()
    DEADLINE = 82.0

    def elapsed():
        return time.time() - start_time

    from math import isqrt
    import bisect

    # Generate all primes <= 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [p for p in range(n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)

    if n <= 1:
        return primes

    # For each prime index i, states[i] is a list of (last_gap, length, prev_index)
    # Pareto-optimal: sorted by last_gap ascending, length strictly increasing
    states = [[] for _ in range(n)]
    states[0].append((0, 1, -1))

    best_length = 1
    best_end = 0

    for i in range(1, n):
        if elapsed() > DEADLINE:
            break

        candidates = [(0, 1, -1)]  # just primes[i] alone
        pi = primes[i]

        for j in range(i):
            if elapsed() > DEADLINE:
                break
            gap = pi - primes[j]
            sj = states[j]
            if not sj:
                continue

            # Binary search: rightmost entry in sj with last_gap <= gap
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 0:
                last_gap_j, len_j, prev_j = sj[lo - 1]
                candidates.append((gap, len_j + 1, j))

        # Filter to Pareto-optimal
        if candidates:
            candidates.sort(key=lambda x: (x[0], -x[1]))
            filtered = []
            max_len = 0
            for gap, length, prev in candidates:
                if length > max_len:
                    filtered.append((gap, length, prev))
                    max_len = length
            states[i] = filtered

            if filtered:
                last_len = filtered[-1][1]
                if last_len > best_length:
                    best_length = last_len
                    best_end = i

    # Reconstruct the sequence ending at best_end
    if best_length <= 1:
        return [primes[best_end]]

    seq = []
    idx = best_end
    while idx != -1:
        seq.append(primes[idx])
        sj = states[idx]
        if not sj:
            break
        # The best entry at idx is the one with max length (last in filtered list)
        _, _, prev = sj[-1]
        idx = prev
    seq.reverse()

    # Local search refinement
    if elapsed() < DEADLINE - 8:
        def is_valid(s):
            if len(s) < 2:
                return True
            for k in range(1, len(s)):
                if s[k] - s[k-1] < s[k-1] - s[k-2] if k >= 2 else False:
                    return False
            # Check nondecreasing gaps
            gaps = [s[k+1] - s[k] for k in range(len(s)-1)]
            for k in range(1, len(gaps)):
                if gaps[k] < gaps[k-1]:
                    return False
            return True

        def greedy_extend_fwd(s):
            """Extend forward: pick smallest valid next prime."""
            ext = list(s)
            while elapsed() < DEADLINE - 8:
                if len(ext) >= 2:
                    last_gap = ext[-1] - ext[-2]
                else:
                    last_gap = 0
                target = ext[-1] + last_gap
                pos = bisect.bisect_left(primes, target)
                if pos < n:
                    ext.append(primes[pos])
                else:
                    break
            return ext

        def greedy_extend_bwd(s):
            """Extend backward: pick largest valid previous prime."""
            ext = list(s)
            while elapsed() < DEADLINE - 8:
                if len(ext) >= 2:
                    first_gap = ext[1] - ext[0]
                else:
                    first_gap = 0
                max_p = ext[0] - first_gap
                if max_p < 2:
                    break
                pos = bisect.bisect_right(primes, max_p) - 1
                if pos >= 0:
                    ext.insert(0, primes[pos])
                else:
                    break
            return ext

        def try_insert(s):
            """Try to insert a prime between consecutive elements."""
            ext = list(s)
            improved = True
            while improved and elapsed() < DEADLINE - 8:
                improved = False
                for pos in range(len(ext) + 1):
                    if elapsed() > DEADLINE - 8:
                        break
                    if pos == 0:
                        # Insert before first
                        if len(ext) >= 2:
                            first_gap = ext[1] - ext[0]
                            max_p = ext[0] - first_gap
                        else:
                            max_p = ext[0] - 1
                        if max_p >= 2:
                            pidx = bisect.bisect_right(primes, max_p) - 1
                            if pidx >= 0 and primes[pidx] < ext[0]:
                                ext.insert(0, primes[pidx])
                                improved = True
                                break
                    elif pos == len(ext):
                        # Insert after last
                        if len(ext) >= 2:
                            last_gap = ext[-1] - ext[-2]
                            target = ext[-1] + last_gap
                        else:
                            target = ext[-1] + 1
                        pidx = bisect.bisect_left(primes, target)
                        if pidx < n:
                            ext.append(primes[pidx])
                            improved = True
                            break
                    else:
                        # Insert between ext[pos-1] and ext[pos]
                        a = ext[pos - 1]
                        b = ext[pos]
                        prev_gap = ext[pos - 1] - ext[pos - 2] if pos >= 2 else 0
                        next_gap = ext[pos + 1] - ext[pos] if pos + 1 < len(ext) else 10000
                        # Need: p - a >= prev_gap, b - p >= p - a, b - p <= next_gap
                        lo_p = max(a + prev_gap, b - next_gap, a + 1)
                        hi_p = (a + b) // 2
                        if lo_p <= hi_p:
                            pidx = bisect.bisect_left(primes, lo_p)
                            if pidx < n and primes[pidx] <= hi_p:
                                ext.insert(pos, primes[pidx])
                                improved = True
                                break
            return ext

        best_seq = list(seq)
        best_len = len(seq)

        # Try greedy extensions in all directions
        for fn in [greedy_extend_fwd, greedy_extend_bwd]:
            ext = fn(seq)
            if len(ext) > best_len:
                best_len = len(ext)
                best_seq = ext

        # Try insertion polish
        if elapsed() < DEADLINE - 10:
            polished = try_insert(best_seq)
            if len(polished) > best_len:
                best_len = len(polished)
                best_seq = polished

        # Try from other high-scoring endpoints
        if elapsed() < DEADLINE - 15:
            # Collect endpoints with good DP scores
            endpoints = []
            for i in range(n):
                if states[i]:
                    ml = states[i][-1][1]
                    if ml >= best_len - 2:
                        endpoints.append((ml, i))
            endpoints.sort(key=lambda x: -x[0])

            tried = 0
            for ml, i in endpoints:
                if elapsed() > DEADLINE - 10 or tried >= 20:
                    break
                # Reconstruct sequence from endpoint i
                si = []
                idx = i
                while idx != -1:
                    si.append(primes[idx])
                    sj = states[idx]
                    if not sj:
                        break
                    _, _, prev = sj[-1]
                    idx = prev
                si.reverse()
                tried += 1

                for fn in [greedy_extend_fwd, greedy_extend_bwd]:
                    ext = fn(si)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best_seq = ext

                if elapsed() < DEADLINE - 10:
                    polished = try_insert(best_seq)
                    if len(polished) > best_len:
                        best_len = len(polished)
                        best_seq = polished

        if len(best_seq) > len(seq):
            seq = best_seq

    return seq
