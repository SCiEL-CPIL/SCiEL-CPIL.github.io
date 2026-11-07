"""Rewritten: maximize |S| subset of 1..128 with no distinct a,b,c in S
having a+b, a+c, b+c all in S. Uses precomputed forbidden triples and
multi-start local search with a wall-clock deadline."""
import time
import random

N = 128
DEADLINE_SEC = 78.0

# Forbidden triple: distinct a<b<c with b+c <= N (so all three sums can fit).
_TRIPLES = []
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        bc_min = b + (b + 1)
        if bc_min > N:
            break
        for c in range(b + 1, N + 1):
            if b + c > N:
                break
            _TRIPLES.append((a, b, c))

# For fast validity: precompute for each triple the three sum indices.
_TRIPLE_SUMS = [
    (a + b, a + c, b + c) for (a, b, c) in _TRIPLES
]


def _valid(mask):
    for i, (a, b, c) in enumerate(_TRIPLES):
        if (mask >> (a - 1)) & 1:
            if (mask >> (b - 1)) & 1:
                if (mask >> (c - 1)) & 1:
                    s1, s2, s3 = _TRIPLE_SUMS[i]
                    if (mask >> (s1 - 1)) & 1 and (mask >> (s2 - 1)) & 1 and (mask >> (s3 - 1)) & 1:
                        return False
    return True


def _popcount(mask):
    return bin(mask).count('1')


def _upper_half_mask():
    m = 0
    for i in range(N // 2 + 1, N + 1):
        m |= 1 << (i - 1)
    return m


def _local_search(start, deadline, rng):
    cur = start
    cur_size = _popcount(cur)
    best = cur
    best_size = cur_size
    # Only accept size-increasing moves; use random restarts externally.
    while time.time() < deadline:
        # try adds
        added = False
        order = list(range(1, N + 1))
        rng.shuffle(order)
        for i in order:
            bit = 1 << (i - 1)
            if cur & bit:
                continue
            if _valid(cur | bit):
                cur |= bit
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best = cur
                added = True
                break
        if not added:
            break
    return best, best_size


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_SEC
    rng = random.Random(20240517)

    base = _upper_half_mask()
    best = base
    best_size = _popcount(base)

    # Start 0: pure upper half, greedy add
    b, s = _local_search(base, deadline, rng)
    if s > best_size:
        best, best_size = b, s

    # Additional starts: upper half + random lower perturbations
    for _ in range(40):
        if time.time() >= deadline:
            break
        m = base
        # add random lower elements
        for _ in range(rng.randint(2, 12)):
            i = rng.randint(1, N // 2)
            m |= 1 << (i - 1)
        # remove a few random upper to allow different structure
        for _ in range(rng.randint(0, 4)):
            up = [i for i in range(N // 2 + 1, N + 1) if (m >> (i - 1)) & 1]
            if up:
                m &= ~(1 << (rng.choice(up) - 1))
        b, s = _local_search(m, deadline, rng)
        if s > best_size:
            best, best_size = b, s

    # Final single-add polish
    for i in range(1, N + 1):
        bit = 1 << (i - 1)
        if best & bit:
            continue
        if _valid(best | bit):
            best |= bit
            best_size += 1

    result = [i for i in range(1, N + 1) if (best >> (i - 1)) & 1]
    return list(result)
