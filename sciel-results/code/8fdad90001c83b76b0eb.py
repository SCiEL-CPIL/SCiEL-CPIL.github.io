"""Max no-two-other-product-divisibility set via bitset local search."""
import time
import random

N = 256
CANDS = list(range(2, N + 1))
M = len(CANDS)
IDX = {v: i for i, v in enumerate(CANDS)}
BIT = [1 << i for i in range(M)]

def _precompute():
    """Precompute div_by[a][b] = bitmask of c such that a | b*c."""
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

def _popcount(S):
    return bin(S).count('1')

def _members(S):
    return [CANDS[i] for i in range(M) if S & BIT[i]]

def _can_add_fast(S, x):
    """Check if x can be added to set S without violating constraints.
    For each a in S: a must not divide x*c for any c in S (c != a, x)
    and x must not divide a*c for any c in S (c != a, x)
    """
    if S & BIT[x]:
        return False
    dx = _DIV_BY[x]
    # For each a in S, check if a | x*c for some c in S\{a,x}
    # div_by[a][x] gives all c such that a | x*c
    # We need: div_by[a][x] & S & ~BIT[a] & ~BIT[x] == 0
    # Also: div_by[x][a] & S & ~BIT[a] & ~BIT[x] == 0
    for a in range(M):
        ba = BIT[a]
        if not (S & ba):
            continue
        if (_DIV_BY[a][x] & S & ~ba & ~BIT[x]):
            return False
        if (dx[a] & S & ~ba & ~BIT[x]):
            return False
    return True

def _local_add(S):
    """Repeatedly add any element that can be added."""
    changed = True
    while changed:
        changed = False
        for x in range(M):
            if not (S & BIT[x]) and _can_add_fast(S, x):
                S |= BIT[x]
                changed = True
                break
    return S

def _greedy(order):
    S = 0
    for x in order:
        if _can_add_fast(S, x):
            S |= BIT[x]
    return S

def _conflict_count(S, x):
    """Count how many constraints x would violate if added to S."""
    if S & BIT[x]:
        return 0
    dx = _DIV_BY[x]
    count = 0
    for a in range(M):
        ba = BIT[a]
        if not (S & ba):
            continue
        count += bin(_DIV_BY[a][x] & S & ~ba & ~BIT[x]).count('1')
        count += bin(dx[a] & S & ~ba & ~BIT[x]).count('1')
    return count

def _swap_improve(S, rng, rounds=8):
    """Try 1-1 swaps that keep size but may enable additions."""
    for _ in range(rounds):
        mems = [i for i in range(M) if S & BIT[i]]
        if not mems:
            break
        y = rng.choice(mems)
        S1 = S & ~BIT[y]
        best_add = None
        best_conf = float('inf')
        for x in range(M):
            if x == y or (S1 & BIT[x]):
                continue
            if _can_add_fast(S1, x):
                best_add = x
                break
        if best_add is None:
            # Try to find best swap by conflict count
            for x in range(M):
                if x == y or (S1 & BIT[x]):
                    continue
                c = _conflict_count(S1, x)
                if c < best_conf:
                    best_conf = c
                    best_add = x
        if best_add is not None:
            S2 = S1 | BIT[best_add]
            S2 = _local_add(S2)
            if _popcount(S2) >= _popcount(S):
                S = S2
    return S

def _two_exchange(S, rng, trials=150):
    """Remove one, add two."""
    best = S
    best_sz = _popcount(S)
    for _ in range(trials):
        mems = [i for i in range(M) if S & BIT[i]]
        if not mems:
            break
        y = rng.choice(mems)
        S1 = S & ~BIT[y]
        cands = []
        for x1 in range(M):
            if x1 == y or (S1 & BIT[x1]):
                continue
            if _can_add_fast(S1, x1):
                cands.append(x1)
        rng.shuffle(cands)
        for x1 in cands[:5]:
            S2 = S1 | BIT[x1]
            for x2 in range(M):
                if x2 == y or x2 == x1 or (S2 & BIT[x2]):
                    continue
                if _can_add_fast(S2, x2):
                    S3 = S2 | BIT[x2]
                    S3 = _local_add(S3)
                    if _popcount(S3) > best_sz:
                        best = S3
                        best_sz = _popcount(S3)
                    break
    return best

def _two_two_exchange(S, rng, trials=80):
    """Remove two, add two - allows restructuring that 1-1/1-2 can't."""
    best = S
    best_sz = _popcount(S)
    for _ in range(trials):
        mems = [i for i in range(M) if S & BIT[i]]
        if len(mems) < 2:
            break
        y1, y2 = rng.sample(mems, 2)
        S1 = S & ~BIT[y1] & ~BIT[y2]
        # Find pairs of addable elements
        addable = []
        for x in range(M):
            if (S1 & BIT[x]) or x == y1 or x == y2:
                continue
            if _can_add_fast(S1, x):
                addable.append(x)
        rng.shuffle(addable)
        for i in range(min(len(addable), 6)):
            x1 = addable[i]
            S2 = S1 | BIT[x1]
            for j in range(min(len(addable), 6)):
                if j == i:
                    continue
                x2 = addable[j]
                if (S2 & BIT[x2]):
                    continue
                if _can_add_fast(S2, x2):
                    S3 = S2 | BIT[x2]
                    S3 = _local_add(S3)
                    if _popcount(S3) > best_sz:
                        best = S3
                        best_sz = _popcount(S3)
                    break
    return best

def _three_exchange(S, rng, trials=40):
    """Remove three, add three - larger restructuring."""
    best = S
    best_sz = _popcount(S)
    for _ in range(trials):
        mems = [i for i in range(M) if S & BIT[i]]
        if len(mems) < 3:
            break
        y1, y2, y3 = rng.sample(mems, 3)
        S1 = S & ~BIT[y1] & ~BIT[y2] & ~BIT[y3]
        addable = []
        for x in range(M):
            if (S1 & BIT[x]) or x in (y1, y2, y3):
                continue
            if _can_add_fast(S1, x):
                addable.append(x)
        rng.shuffle(addable)
        if len(addable) >= 3:
            x1, x2, x3 = addable[:3]
            S2 = S1 | BIT[x1]
            if _can_add_fast(S2, x2):
                S2 |= BIT[x2]
                if _can_add_fast(S2, x3):
                    S2 |= BIT[x3]
                    S2 = _local_add(S2)
                    if _popcount(S2) > best_sz:
                        best = S2
                        best_sz = _popcount(S2)
    return best

def _value_order(S):
    """Order candidates by value (higher values first) for greedy."""
    return sorted(range(M), key=lambda i: -CANDS[i])

def _degree_order(S):
    """Order candidates by number of conflicts with S (fewer first)."""
    conflicts = []
    for x in range(M):
        if S & BIT[x]:
            continue
        c = _conflict_count(S, x)
        conflicts.append((c, x))
    conflicts.sort()
    return [x for _, x in conflicts]

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

    # Try value-based greedy as another starting point
    val_order = _value_order(0)
    S_val = _greedy(val_order)
    S_val = _local_add(S_val)
    if _popcount(S_val) > best:
        best = _popcount(S_val)
        best_S = S_val

    # Try degree-based greedy
    S_deg = _greedy(_degree_order(0))
    S_deg = _local_add(S_deg)
    if _popcount(S_deg) > best:
        best = _popcount(S_deg)
        best_S = S_deg

    for seed in range(300):
        if time.time() > deadline - 3:
            break
        rng = random.Random(seed)
        
        # Strategy 1: random greedy
        if seed % 4 != 3:
            order = list(range(M))
            rng.shuffle(order)
            S = _greedy(order)
            S = _local_add(S)
            S = _swap_improve(S, rng)
            S = _two_exchange(S, rng)
            S = _two_two_exchange(S, rng)
            S = _three_exchange(S, rng)
            S = _local_add(S)
            sz = _popcount(S)
            if sz > best:
                best = sz
                best_S = S
        
        # Strategy 2: perturb prime set
        if seed % 2 == 0:
            P = prime_S
            mems = [i for i in range(M) if P & BIT[i]]
            k = rng.randint(1, 5)
            for _ in range(k):
                y = rng.choice(mems)
                P &= ~BIT[y]
                mems = [i for i in range(M) if P & BIT[i]]
            P = _local_add(P)
            P = _swap_improve(P, rng)
            P = _two_exchange(P, rng)
            P = _two_two_exchange(P, rng)
            P = _three_exchange(P, rng)
            P = _local_add(P)
            if _popcount(P) > best:
                best = _popcount(P)
                best_S = P
        
        # Strategy 3: perturb best found so far
        if seed % 4 == 3 and best_S != 0:
            B = best_S
            mems = [i for i in range(M) if B & BIT[i]]
            k = rng.randint(1, 4)
            for _ in range(k):
                y = rng.choice(mems)
                B &= ~BIT[y]
                mems = [i for i in range(M) if B & BIT[i]]
            B = _local_add(B)
            B = _swap_improve(B, rng)
            B = _two_exchange(B, rng)
            B = _two_two_exchange(B, rng)
            B = _three_exchange(B, rng)
            B = _local_add(B)
            if _popcount(B) > best:
                best = _popcount(B)
                best_S = B

    # Final polish: try more iterations on best solution
    if time.time() < deadline - 5:
        rng = random.Random(12345)
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            B = best_S
            mems = [i for i in range(M) if B & BIT[i]]
            k = rng.randint(1, 3)
            for _ in range(k):
                y = rng.choice(mems)
                B &= ~BIT[y]
                mems = [i for i in range(M) if B & BIT[i]]
            B = _local_add(B)
            B = _swap_improve(B, rng, rounds=10)
            B = _two_exchange(B, rng, trials=200)
            B = _two_two_exchange(B, rng, trials=100)
            B = _three_exchange(B, rng, trials=60)
            B = _local_add(B)
            if _popcount(B) > best:
                best = _popcount(B)
                best_S = B

    return _members(best_S)
