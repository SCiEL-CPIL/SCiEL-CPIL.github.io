import time

P = {'gaps': [[10007, 10009], [10009, 10037], [100037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_lpf(n):
    """Return array lpf where lpf[i] = least prime factor of i, for i in [0, n]."""
    lpf = list(range(n + 1))
    for i in range(2, int(n**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf


def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    max_prime = max(hi for _, hi in P['gaps'])
    lpf = _sieve_lpf(max_prime)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        found = None
        # Scan from lo+1 to hi-1
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                found = m
                break
        if found is None:
            # Try from the other end as a secondary strategy
            for m in range(hi - 1, lo, -1):
                if lpf[m] >= gap:
                    found = m
                    break
        result[i] = found

    # Ensure all entries are Python int or None
    final_result = []
    for r in result:
        if r is None:
            final_result.append(None)
        else:
            final_result.append(int(r))

    return final_result
