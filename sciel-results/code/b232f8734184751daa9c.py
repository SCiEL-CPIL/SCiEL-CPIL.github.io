import time
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _calc_threshold(a, b):
    img_a = [1 << a[i] for i in range(N)]
    img_b = [1 << b[i] for i in range(N)]
    start = FULL_MASK
    if (start & (start - 1)) == 0:
        return 0
    visited = {start: 0}
    q = deque([start])
    while q:
        curr = q.popleft()
        d = visited[curr]
        if (curr & (curr - 1)) == 0:
            return d
        for img in (img_a, img_b):
            nxt = 0
            t = curr
            while t:
                lsb = t & (-t)
                nxt |= img[lsb.bit_length() - 1]
                t ^= lsb
            if nxt not in visited:
                visited[nxt] = d + 1
                q.append(nxt)
    return -1

def calculate_threshold_fast(a_trans, b_trans):
    return _calc_threshold(a_trans, b_trans)

def construct():
    # Known optimal 12-state Černý automaton (cyclic a, designed b).
    a = [(i + 1) % N for i in range(N)]
    b = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    score = _calc_threshold(a, b)
    if score >= 100:
        return [a, b]

    # Fallback: search for a good synchronizing automaton.
    start = time.time()
    deadline = start + 85.0
    best = 0
    best_a, best_b = a, b

    def try_auto(x, y):
        nonlocal best, best_a, best_b
        s = _calc_threshold(x, y)
        if s > best:
            best = s
            best_a, best_b = x[:], y[:]

    # Structured: cyclic a, b = identity with defects.
    for k in range(N):
        for v in range(N):
            if v == k:
                continue
            bb = list(range(N)); bb[k] = v
            try_auto(a, bb)
    for k1 in range(N):
        for k2 in range(k1+1, N):
            for v1 in range(N):
                for v2 in range(N):
                    if v1 == k1 and v2 == k2:
                        continue
                    bb = list(range(N)); bb[k1] = v1; bb[k2] = v2
                    if len(set(bb)) == N:
                        continue
                    try_auto(a, bb)
                    if time.time() > deadline - 10:
                        return [best_a, best_b]

    # Random search with local improvement.
    import random
    rng = random.Random(7)
    for _ in range(4000):
        if time.time() > deadline - 3:
            break
        x = [rng.randint(0, N-1) for _ in range(N)]
        y = [rng.randint(0, N-1) for _ in range(N)]
        if len(set(x)) == N:
            x[rng.randint(0, N-1)] = x[rng.randint(0, N-1)]
        if len(set(y)) == N:
            y[rng.randint(0, N-1)] = y[rng.randint(0, N-1)]
        try_auto(x, y)
    return [best_a, best_b]
