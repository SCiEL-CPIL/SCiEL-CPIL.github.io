import time
from math import isqrt
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    def primes_upto(n):
        if n < 2:
            return []
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0] = 0
        if n >= 1:
            sieve[1] = 0
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n + 1) if sieve[i]]

    primes = primes_upto(10000)
    m = len(primes)

    if m < 2:
        return primes

    # dp_gaps[j]: sorted list of gaps (unique, increasing)
    # dp_lens[j]: corresponding lengths (strictly increasing)
    # dp_parents[j]: corresponding (prev_index, prev_gap) tuples
    dp_gaps = [[] for _ in range(m)]
    dp_lens = [[] for _ in range(m)]
    dp_parents = [[] for _ in range(m)]

    dp_gaps[0] = [0]
    dp_lens[0] = [1]
    dp_parents[0] = [(-1, 0)]

    best_len = 1
    best_end = 0
    best_gap = 0

    for j in range(1, m):
        if time.time() - t0 > DEADLINE:
            break
        pj = primes[j]
        candidates = []

        for i in range(j):
            d = pj - primes[i]
            gaps_i = dp_gaps[i]
            if not gaps_i:
                continue
            idx = bisect.bisect_right(gaps_i, d) - 1
            if idx >= 0:
                prev_gap = gaps_i[idx]
                prev_len = dp_lens[i][idx]
                new_len = prev_len + 1
                candidates.append((d, new_len, i, prev_gap))

        if not candidates:
            dp_gaps[j] = [0]
            dp_lens[j] = [1]
            dp_parents[j] = [(-1, 0)]
            continue

        candidates.sort(key=lambda x: (x[0], -x[1]))

        frontier_gaps = []
        frontier_lens = []
        frontier_parents = []
        max_len = -1

        for gap, length, prev_i, prev_gap in candidates:
            if length > max_len:
                frontier_gaps.append(gap)
                frontier_lens.append(length)
                frontier_parents.append((prev_i, prev_gap))
                max_len = length

        dp_gaps[j] = frontier_gaps
        dp_lens[j] = frontier_lens
        dp_parents[j] = frontier_parents

        if frontier_lens:
            last_gap = frontier_gaps[-1]
            last_len = frontier_lens[-1]
            if last_len > best_len:
                best_len = last_len
                best_end = j
                best_gap = last_gap

    seq = [primes[best_end]]
    cur_idx = best_end
    cur_gap = best_gap

    idx_in_frontier = bisect.bisect_left(dp_gaps[cur_idx], cur_gap)

    if idx_in_frontier < len(dp_gaps[cur_idx]) and dp_gaps[cur_idx][idx_in_frontier] == cur_gap:
        while True:
            prev_i, prev_gap = dp_parents[cur_idx][idx_in_frontier]
            if prev_i == -1:
                break
            seq.append(primes[prev_i])
            cur_idx = prev_i
            cur_gap = prev_gap
            idx_in_frontier = bisect.bisect_left(dp_gaps[cur_idx], cur_gap)
            if idx_in_frontier >= len(dp_gaps[cur_idx]) or dp_gaps[cur_idx][idx_in_frontier] != cur_gap:
                break
    else:
        pass

    seq.reverse()

    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                values = [primes[0]]
                for p in primes[1:]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
                return values

    return seq
