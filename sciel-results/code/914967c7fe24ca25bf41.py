def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    from math import isqrt
    import numpy as np

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
    primes_arr = np.array(primes, dtype=np.int64)

    if n <= 1:
        return primes

    # For each prime index i, states[i] is a list of (last_gap, length, prev_index)
    # Pareto-optimal: sorted by last_gap ascending, length strictly increasing
    # (0, 1, -1) represents a sequence of length 1 (just primes[i] itself)
    states = [[] for _ in range(n)]
    states[0].append((0, 1, -1))

    best_length = 1
    best_end = 0

    for i in range(1, n):
        if elapsed() > DEADLINE:
            break

        candidates = [(0, 1, -1)]  # just primes[i] alone

        pi = primes[i]
        
        # Vectorized gap computation for all j < i
        gaps = pi - primes_arr[:i]
        
        for j in range(i):
            if elapsed() > DEADLINE:
                break

            gap = int(gaps[j])
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
            # lo-1 is the last valid index (last_gap <= gap)
            if lo > 0:
                last_gap_j, len_j, prev_j = sj[lo - 1]
                candidates.append((gap, len_j + 1, j))

        # Filter to Pareto-optimal: sort by gap, keep strictly increasing length
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
    
    # Local search: try to extend the sequence by inserting primes
    # or finding a better ending
    if elapsed() < DEADLINE - 5:
        seq_set = set(seq)
        current_gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        
        # Try to find a longer sequence by greedy extension from the end
        extended = list(seq)
        while elapsed() < DEADLINE - 5:
            last_gap = extended[-1] - extended[-2] if len(extended) >= 2 else 0
            last_prime = extended[-1]
            # Find next prime > last_prime with gap >= last_gap
            next_prime = None
            for p in primes:
                if p > last_prime and (p - last_prime) >= last_gap:
                    next_prime = p
                    break
            if next_prime is None:
                break
            extended.append(next_prime)
        
        if len(extended) > len(seq):
            seq = extended
            seq_set = set(seq)
            current_gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
    
    return seq
