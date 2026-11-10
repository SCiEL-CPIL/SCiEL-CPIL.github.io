# EVOLVE-BLOCK-START
import time
import random

def _valid_set(s, N):
    """Return True iff no element of s is the arithmetic mean of any
    subset (size >=2) of the OTHER elements.  Uses integer bitsets."""
    n = len(s)
    if n < 3:
        return True
    others = [0] * n
    for i in range(n):
        m = (1 << n) - 1
        m &= ~(1 << i)
        others[i] = m
    for i in range(n):
        a = s[i]
        others_mask = others[i]
        # dp[k] = bitset of achievable sums of exactly k elements from others[i]
        dp = [0] * (n + 1)
        dp[0] = 1
        m = others_mask
        while m:
            low = m & (-m)
            j = (low.bit_length() - 1)
            m ^= low
            v = s[j]
            for k in range(n, 0, -1):
                dp[k] |= dp[k - 1] << v
        for k in range(2, n):
            if (dp[k] >> (k * a)) & 1:
                return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0
    N = 512
    MAXC = 48
    rng = random.Random(20240607)

    best = None
    best_len = 0

    # ---- Phase A: greedy construction (fast bitset feasibility) ----
    def greedy(start):
        sel = []
        for x in range(start, N + 1):
            cand = sel + [x]
            if _valid_set(cand, N):
                sel = cand
                if len(sel) >= MAXC:
                    break
        return sel

    best = greedy(1)
    best_len = len(best)

    # ---- Phase B: local search + simulated annealing to maximize cardinality ----
    def neighbors(sel):
        n = len(sel)
        nb = []
        # add
        if n < MAXC:
            for x in range(1, N + 1):
                if x not in sel:
                    nb.append((x, 'add'))
        # remove
        for i in range(n):
            nb.append((i, 'rm'))
        # swap
        if n:
            for i in range(n):
                for x in range(1, N + 1):
                    if x not in sel:
                        nb.append((i, 'sw', x))
        return nb

    def apply_move(sel, move):
        if move[1] == 'add':
            return sel + [move[0]]
        if move[1] == 'rm':
            i = move[0]
            return sel[:i] + sel[i + 1:]
        i, x = move[0], move[2]
        return sel[:i] + [x] + sel[i + 1:]

    cur = sorted(best)
    cur_len = len(cur)
    T = 4.0
    T_min = 0.05
    cool = 0.985

    while time.time() < DEADLINE:
        if len(cur) > best_len:
            best_len = len(cur)
            best = list(cur)
            T = 4.0
        nb = neighbors(cur)
        rng.shuffle(nb)
        tried = 0
        for move in nb:
            if time.time() > DEADLINE:
                break
            tried += 1
            cand = apply_move(cur, move)
            if not _valid_set(cand, N):
                continue
            cl = len(cand)
            if cl > cur_len:
                cur = cand
                T = 4.0
                break
            elif cl < cur_len:
                d = cl - cur_len
                if rng.random() < pow(T, d):
                    cur = cand
            else:
                # same size: keep (random walk) occasionally
                if rng.random() < 0.2:
                    cur = cand
            if tried > 400:
                break
        T *= cool
        if T < T_min:
            # restart from best with perturbation
            cur = list(best)
            T = 4.0

    return best

# EVOLVE-BLOCK-END
