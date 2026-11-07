"""Max no-two-other-product-divisibility set via bitset local search."""
import time
import random

N = 256
CANDS = list(range(2, N + 1))
M = len(CANDS)
IDX = {v: i for i, v in enumerate(CANDS)}
BIT = [1 << i for i in range(M)]

def _precompute():
    div_by = [[0] * M for _ in range(M)]
    for a in range(M):
        va = CANDS[a]
        row = div_by[a]
        for b in range(M):
            vb = CANDS[b]
            mask = 0
            for c in range(M):
                if (vb * CANDS[c]) % va == 0:
                    mask |= BIT[c]
            row[b] = mask
    return div_by

_DIV_BY = _precompute()
_ALL = (1 << M) - 1

def _can_add(S, x):
    if S & BIT[x]:
        return False
    dx = _DIV_BY[x]
    for a in range(M):
        ba = BIT[a]
        if not (S & ba):
            continue
        if (_DIV_BY[a][x] & S & ~ba):
            return False
        if (dx[a] & S & ~ba):
            return False
    return True

def _popcount(S):
    return bin(S).count('1')

def _members(S):
    return [CANDS[i] for i in range(M) if S & BIT[i]]

def _local_add(S):
    changed = True
    while changed:
        changed = False
        for x in range(M):
            if not (S & BIT[x]) and _can_add(S, x):
                S |= BIT[x]
                changed = True
                break
    return S

def _greedy(order):
    S = 0
    for x in order:
        if _can_add(S, x):
            S |= BIT[x]
    return S

def _swap_improve(S, rng, rounds=5):
    # try 1-1 swaps that keep size but may enable additions, then local_add
    for _ in range(rounds):
        mems = [i for i in range(M) if S & BIT[i]]
        if not mems:
            break
        y = rng.choice(mems)
        S1 = S & ~BIT[y]
        best_add = None
        for x in range(M):
            if x == y or (S1 & BIT[x]):
                continue
            if _can_add(S1, x):
                best_add = x
                break
        if best_add is not None:
            S2 = S1 | BIT[best_add]
            S2 = _local_add(S2)
            if _popcount(S2) >= _popcount(S):
                S = S2
    return S

def _two_exchange(S, rng, trials=100):
    # remove one, add two
    best = S
    best_sz = _popcount(S)
    for _ in range(trials):
        mems = [i for i in range(M) if S & BIT[i]]
        if not mems:
            break
        y = rng.choice(mems)
        S1 = S & ~BIT[y]
        # find two addable
        cands = []
        for x1 in range(M):
            if x1 == y or (S1 & BIT[x1]):
                continue
            if _can_add(S1, x1):
                cands.append(x1)
        rng.shuffle(cands)
        for x1 in cands[:3]:
            S2 = S1 | BIT[x1]
            for x2 in range(M):
                if x2 == y or x2 == x1 or (S2 & BIT[x2]):
                    continue
                if _can_add(S2, x2):
                    S3 = S2 | BIT[x2]
                    S3 = _local_add(S3)
                    if _popcount(S3) > best_sz:
                        best = S3
                        best_sz = _popcount(S3)
                    break
    return best

def construct():
    deadline = time.time() + 80.0
    best_S = 0
    best = 0

    # baseline: primes
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True
    prime_S = 0
    for v in CANDS:
        if is_prime(v):
            prime_S |= BIT[IDX[v]]
    best_S = prime_S
    best = _popcount(prime_S)

    for seed in range(200):
        if time.time() > deadline - 3:
            break
        rng = random.Random(seed)
        # random greedy
        order = list(range(M))
        rng.shuffle(order)
        S = _greedy(order)
        S = _local_add(S)
        S = _swap_improve(S, rng)
        S = _two_exchange(S, rng)
        S = _local_add(S)
        sz = _popcount(S)
        if sz > best:
            best = sz
            best_S = S
        # also perturb prime set
        if seed % 2 == 0:
            P = prime_S
            mems = [i for i in range(M) if P & BIT[i]]
            k = rng.randint(1, 4)
            for _ in range(k):
                y = rng.choice(mems)
                P &= ~BIT[y]
                mems = [i for i in range(M) if P & BIT[i]]
            P = _local_add(P)
            P = _swap_improve(P, rng)
            P = _two_exchange(P, rng)
            P = _local_add(P)
            if _popcount(P) > best:
                best = _popcount(P)
                best_S = P
    return _members(best_S)
