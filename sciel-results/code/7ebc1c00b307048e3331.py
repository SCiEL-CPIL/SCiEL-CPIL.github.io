import time
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85

    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        if n >= 0:
            is_prime[0] = 0
        if n >= 1:
            is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = 0
        return [p for p in range(2, n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)

    if n <= 1:
        return primes

    # states[i] is a Pareto front of (last_gap, length, prev_idx)
    # sorted by last_gap ascending; lengths strictly increasing along the front.
    states = [None] * n
    states[0] = [(0, 1, -1)]

    best_length = 1
    best_end = 0

    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break

        pi = primes[i]
        candidates = [(0, 1, -1)]

        for j in range(i):
            if time.time() - start_time > DEADLINE:
                break

            sj = states[j]
            if not sj:
                continue

            gap = pi - primes[j]
            # Binary search for largest entry with last_gap <= gap
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid

            if lo > 0:
                prev_gap, prev_len, prev_idx = sj[lo - 1]
                candidates.append((gap, prev_len + 1, j))

        # Build Pareto front for i
        if candidates:
            candidates.sort(key=lambda x: (x[0], -x[1]))
            filtered = []
            max_len_so_far = 0
            for gap, length, prev_idx in candidates:
                if length > max_len_so_far:
                    filtered.append((gap, length, prev_idx))
                    max_len_so_far = length
            states[i] = filtered

            if filtered:
                last_len = filtered[-1][1]
                if last_len > best_length:
                    best_length = last_len
                    best_end = i

    # Reconstruct
    if best_length <= 1:
        return [primes[best_end]]

    target_front = states[best_end]
    if not target_front:
        return [primes[best_end]]

    # Find state with length == best_length
    prev_idx = -1
    for gap, length, p_idx in target_front:
        if length == best_length:
            prev_idx = p_idx
            break

    if prev_idx == -1:
        prev_idx = target_front[-1][2]

    seq = [primes[best_end]]
    idx = best_end
    current_len = best_length
    while current_len > 1:
        front_at_idx = states[idx]
        if not front_at_idx:
            break
        found = False
        for gap, length, p_idx in front_at_idx:
            if length == current_len:
                seq.append(primes[p_idx])
                idx = p_idx
                current_len -= 1
                found = True
                break
        if not found:
            break

    seq.reverse()

    # Verification
    def is_prime_check(p):
        if p < 2:
            return False
        if p == 2:
            return True
        if p % 2 == 0:
            return False
        for d in range(3, isqrt(p) + 1, 2):
            if p % d == 0:
                return False
        return True

    is_valid = True
    for k in range(len(seq)):
        if not is_prime_check(seq[k]) or seq[k] > 10000:
            is_valid = False
            break
    if is_valid:
        for k in range(len(seq) - 1):
            if seq[k] >= seq[k + 1]:
                is_valid = False
                break
    if is_valid:
        for k in range(2, len(seq)):
            gap_curr = seq[k] - seq[k - 1]
            gap_prev = seq[k - 1] - seq[k - 2]
            if gap_curr < gap_prev:
                is_valid = False
                break

    if not is_valid:
        return [primes[0]]

    return seq
