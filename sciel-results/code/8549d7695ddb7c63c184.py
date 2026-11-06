"""Smooth two-summand decomposition of 123457."""
import time

TARGET = 123457

def _largest_prime_factor(n):
    if n <= 1:
        return 1
    p = 2
    m = n
    while p * p <= m:
        while m % p == 0:
            m //= p
        p += 1 if p == 2 else 2
    return m if m > 1 else p

def _generate_smooth(B, limit):
    # all positive integers <= limit whose prime factors are all <= B
    primes = []
    x = 2
    while x <= B:
        is_p = True
        for d in primes:
            if d * d > x:
                break
            if x % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
        x += 1
    # BFS over exponents (B is small so the set is tiny)
    result = {1}
    changed = True
    while changed:
        changed = False
        for v in list(result):
            for p in primes:
                nv = v * p
                if nv > limit:
                    break
                if nv not in result:
                    result.add(nv)
                    changed = True
    return result

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin under 90s

    best = (1, TARGET - 1)
    best_val = max(_largest_prime_factor(best[0]), _largest_prime_factor(best[1]))

    for B in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if time.time() > deadline:
            break
        smooth = _generate_smooth(B, TARGET)
        # find a in smooth with complement also smooth
        found = None
        for a in smooth:
            b = TARGET - a
            if b >= 1 and b in smooth:
                found = (a, b)
                break
        if found is None:
            continue
        # verify exact value for this candidate; keep best
        val = max(_largest_prime_factor(found[0]), _largest_prime_factor(found[1]))
        if val < best_val:
            best = found
            best_val = val
        # if we reached a very small bound the value is likely optimal; stop early
        if best_val <= B:
            break

    return list(best)
