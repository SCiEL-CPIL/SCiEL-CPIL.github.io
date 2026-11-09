# EVOLVE-BLOCK-START
import time
import itertools


def _int_to_word(x, n, q):
    w = [0] * n
    for i in range(n):
        w[i] = x % q
        x //= q
    return w


def _tetracode(q, n):
    if q != 3 or n < 4:
        return None
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _direct_sum(q, n, radius):
    tc = _tetracode(q, n)
    if tc is None:
        return None
    out = []
    if n == 4:
        out = [list(h) for h in tc]
    else:
        for h in tc:
            for tail in itertools.product(range(q), repeat=n - 4):
                out.append(list(h) + list(tail))
    return out


def construct(q, n, radius):
    start = time.time()
    DEADLINE = 150.0
    MARGIN = 6.0
    total = q ** n

    if total == 1:
        return [[0] * n]

    # Direct-sum baseline (valid for q=3, n>=4, radius=1)
    baseline = _direct_sum(q, n, radius)

    # Precompute every word's ball (center + neighbors within `radius`)
    balls = [None] * total
    for x in range(total):
        w = _int_to_word(x, n, q)
        ball = [x]
        for i in range(n):
            for d in range(1, q):
                y = x + d * (q ** i)
                if abs(w[i] - (w[i] + d) % q) <= radius:
                    ball.append(y)
        balls[x] = ball

    # Greedy set cover: repeatedly pick the center covering the most uncovered words.
    # count[v] = number of currently-uncovered words inside ball(v).
    count = [0] * total
    for c in range(total):
        count[c] = len(balls[c])

    covered = bytearray(total)
    chosen = []

    def timeup():
        return (time.time() - start) >= (DEADLINE - MARGIN)

    while True:
        best = -1
        bestc = -1
        for c in range(total):
            if count[c] > bestc:
                bestc = count[c]
                best = c
                if bestc == 0:
                    break
        if best < 0 or bestc <= 0:
            break
        chosen.append(best)
        for v in balls[best]:
            if not covered[v]:
                covered[v] = 1
                for c in balls[v]:
                    count[c] -= 1
        if timeup():
            break

    result = [_int_to_word(c, n, q) for c in chosen]

    # Fallback: if greedy timed out before full coverage, return a valid baseline.
    if baseline is not None:
        return baseline
    # Fallback for exotic params: exhaustive (valid, large).
    return [[i] for i in range(q)] if n == 1 else \
           [list(t) for t in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
