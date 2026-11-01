"""Football pool K_3(6,1): find a small ternary covering code of length 6, radius 1.

Strategy: greedy set-cover over the seed's 81-word candidate pool.  Each candidate
covers its radius-1 Hamming ball (28 words).  Repeatedly pick the candidate that
covers the most not-yet-covered words; stop when all 729 words are covered.  A
deadline guard returns the best valid cover found before the wall-clock limit.
"""
import time
import itertools

_DEADLINE = 150.0
_MARGIN = 5.0


def _ball_mask(c, q, n):
    """Bitmask of the radius-1 Hamming ball around word c (c plus 1-neighborhood)."""
    mask = 1 << c
    for i in range(n):
        base = c // (q ** (n - 1 - i))
        for d in range(1, q):
            mask |= 1 << (c + d * (q ** (n - 1 - i)))
    return mask


def construct(q, n, radius):
    t0 = time.time()
    limit = _DEADLINE - _MARGIN

    # Candidate pool: direct sum of the ternary tetracode with every tail word.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q]
                 for a in range(q) for b in range(q)]
    candidates = [tuple(h + list(tail))
                  for h in tetracode
                  for tail in itertools.product(range(q), repeat=n - 4)]

    # Deduplicate identical radius-1 balls (keep one representative per ball).
    seen = {}
    for c in candidates:
        seen.setdefault(_ball_mask(c, q, n), c)
    reps = list(seen.values())

    # Precompute coverage bitmasks for each representative.
    cov = [_ball_mask(c, q, n) for c in reps]
    m = len(cov)
    total = q ** n
    full = (1 << total) - 1

    covered = 0
    chosen = []
    in_use = [True] * m
    while covered < total:
        best_j = -1
        best_gain = 0
        for j in range(m):
            if not in_use[j]:
                continue
            g = bin(covered & cov[j]).count("1")
            if g > best_gain:
                best_gain = g
                best_j = j
        if best_j < 0:
            break
        chosen.append(reps[best_j])
        covered |= cov[best_j]
        in_use[best_j] = False
        if time.time() > limit:
            break

    # Build the code from the chosen representatives.
    code = [list(c) for c in chosen]

    # If we ran out of time before full coverage, extend greedily from the full
    # space until valid (this only triggers on a timeout).
    if covered < total:
        remaining = list(range(total))
        while covered < total:
            best_v = -1
            best_gain = 0
            for v in remaining:
                digits = []
                x = v
                for i in range(n - 1, -1, -1):
                    digits.append(x % q)
                    x //= q
                digits = digits[::-1]
                g = bin(covered & _ball_mask(sum(d * q ** (n - 1 - i) for i, d in enumerate(digits)), q, n)).count("1")
                if g > best_gain:
                    best_gain = g
                    best_v = v
            if best_v < 0:
                break
            digits = []
            x = best_v
            for i in range(n - 1, -1, -1):
                digits.append(x % q)
                x //= q
            code.append(digits[::-1])
            covered |= _ball_mask(best_v, q, n)
            remaining.remove(best_v)
            if time.time() > _DEADLINE:
                break

    return code
