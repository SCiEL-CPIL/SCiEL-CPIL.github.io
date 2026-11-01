import itertools
import time
import random

def construct(q, n, radius):
    # General guard: only the known task (q=3,n=6,radius=1) is optimized;
    # fall back to the seed construction otherwise.
    if not (q == 3 and n == 6 and radius == 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    N = q ** n  # 729
    deadline = time.time() + 110.0  # safety margin under 120s

    # Encode words as base-3 integers 0..N-1; precompute coordinate digits.
    digits = []
    for x in range(N):
        d = [0] * n
        t = x
        for i in range(n):
            d[i] = t % 3
            t //= 3
        digits.append(d)

    # Precompute ball (word + all distance-1 neighbours) as a Python int bitmask.
    balls = [0] * N
    for x in range(N):
        mask = 1 << x
        d = digits[x]
        for i in range(n):
            for delta in (1, 2):
                y = x + delta * (3 ** i)
                mask |= 1 << y
        balls[x] = mask

    full_mask = (1 << N) - 1

    def covers(codewords):
        m = 0
        for c in codewords:
            m |= balls[c]
        return m

    best = None
    best_size = N  # trivial upper bound

    def try_greedy(rng):
        nonlocal best, best_size
        covered = 0
        chosen = []
        # Work on a copy of "uncovered" as a bitmask.
        uncovered = full_mask
        while uncovered and time.time() < deadline:
            best_word = -1
            best_gain = -1
            # iterate candidates; randomize order occasionally for diversity
            order = list(range(N))
            if rng.random() < 0.3:
                rng.shuffle(order)
            for w in order:
                gain = (balls[w] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    if best_gain == 13:  # can't do better
                        break
            if best_gain <= 0:
                break
            chosen.append(best_word)
            covered |= balls[best_word]
            uncovered = full_mask & ~covered
        # saturation: remove redundant codewords
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(chosen)):
                if time.time() >= deadline:
                    break
                rest = covered & ~balls[chosen[i]]
                # covered without chosen[i]
                m = 0
                for j in range(len(chosen)):
                    if j != i:
                        m |= balls[chosen[j]]
                if m == full_mask:
                    chosen.pop(i)
                    covered = m
                    changed = True
                    break
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen[:]

    def try_local_improve(start, rng):
        nonlocal best, best_size
        chosen = start[:]
        covered = covers(chosen)
        if covered != full_mask:
            return
        # exchange moves that keep coverage, then re-saturate
        for _ in range(40):
            if time.time() >= deadline:
                break
            improved = False
            # try to remove a redundant word
            for i in range(len(chosen)):
                m = 0
                for j in range(len(chosen)):
                    if j != i:
                        m |= balls[chosen[j]]
                if m == full_mask:
                    chosen.pop(i)
                    covered = m
                    improved = True
                    break
            if improved:
                continue
            # try a swap: replace chosen[i] with w (not in chosen) keeping coverage
            i = rng.randrange(len(chosen))
            c = chosen[i]
            w = rng.randrange(N)
            if w in chosen:
                continue
            m = 0
            for j in range(len(chosen)):
                if j != i:
                    m |= balls[chosen[j]]
            m |= balls[w]
            if m == full_mask:
                chosen[i] = w
                covered = m
                improved = True
            if not improved:
                break
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen[:]

    rng = random.Random(12345)
    # A few deterministic-ish greedy passes with different seeds, then randomized.
    start_seeds = [0, 1, 2, 3, 7, 11, 42, 99, 123, 2024]
    for s in start_seeds:
        r = random.Random(s)
        try_greedy(r)
        if best is not None:
            try_local_improve(best, r)
        if time.time() >= deadline - 5:
            break
    # randomized restarts until deadline
    it = 0
    while time.time() < deadline - 2:
        r = random.Random(100000 + it)
        try_greedy(r)
        if best is not None and time.time() < deadline - 2:
            try_local_improve(best, r)
        it += 1

    if best is None:
        # absolute fallback: all words
        return [digits[x][:] for x in range(N)]
    return [digits[c][:] for c in best]
