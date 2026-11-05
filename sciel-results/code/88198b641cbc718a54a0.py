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

    # Fix: reconstruct by tracking the actual best-length path
    # We need to store for each (i, best_length_at_i) which prev index was used
    # Since states[i] is sorted by gap with strictly increasing length,
    # the entry with the max length is the last one, but we need to verify
    # that following prev actually gives us the correct length.
    
    # Better approach: rebuild by finding a valid path
    seq = []
    idx = best_end
    remaining_len = best_length
    
    while idx != -1 and remaining_len > 0:
        seq.append(primes[idx])
        if remaining_len == 1:
            break
        
        sj = states[idx]
        # Find the entry in sj with length == remaining_len - 1
        # and verify the gap constraint is satisfied
        found_prev = None
        for gap_j, len_j, prev_j in sj:
            if len_j == remaining_len - 1:
                # Check gap constraint: primes[idx] - primes[prev_j] >= gap_j
                if primes[idx] - primes[prev_j] >= gap_j:
                    found_prev = prev_j
                    break
            # If len_j > remaining_len - 1, no point continuing (sorted by gap, length increases)
            # Actually states are sorted by gap, not length. Let's just scan.
        
        if found_prev is None:
            # Fallback: try the last entry
            if sj:
                found_prev = sj[-1][2]
        
        if found_prev is None:
            break
        idx = found_prev
        remaining_len -= 1

    seq.reverse()
    
    # Multi-start local search: try to extend from multiple positions
    # and try different ending primes
    if elapsed() < DEADLINE - 5:
        best_seq = list(seq)
        
        # Try greedy extension from the end of our current sequence
        extended = list(seq)
        while elapsed() < DEADLINE - 5:
            if len(extended) < 2:
                break
            last_gap = extended[-1] - extended[-2]
            last_prime = extended[-1]
            next_prime = None
            for p in primes:
                if p > last_prime and (p - last_prime) >= last_gap:
                    next_prime = p
                    break
            if next_prime is None:
                break
            extended.append(next_prime)
        
        if len(extended) > len(best_seq):
            best_seq = extended
        
        # Try to find a better sequence by scanning all primes as potential starts
        # and doing greedy forward extension with nondecreasing gaps
        if elapsed() < DEADLINE - 10:
            for start_idx in range(n):
                if elapsed() > DEADLINE - 5:
                    break
                # Start with primes[start_idx], gap starts at 0
                candidate = [primes[start_idx]]
                current_gap = 0
                for next_idx in range(start_idx + 1, n):
                    if elapsed() > DEADLINE - 5:
                        break
                    gap = primes[next_idx] - primes[-1]
                    if gap >= current_gap:
                        candidate.append(primes[next_idx])
                        current_gap = gap
                
                if len(candidate) > len(best_seq):
                    best_seq = candidate
        
        seq = best_seq
    
    return seq
