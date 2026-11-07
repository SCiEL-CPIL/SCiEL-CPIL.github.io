import time
import random

N = 256

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

LARGE_PRIMES = [p for p in range(129, N + 1) if is_prime(p)]
CANDS = [x for x in range(2, N + 1) if x not in LARGE_PRIMES]
M = len(CANDS)
BIT = [1 << i for i in range(M)]

def _precompute():
    div_by = [[0] * M for _ in range(M)]
    divs_of_prod = [[0] * M for _ in range(M)]
    div_mask = [0] * M
    mult_mask = [0] * M
    
    for a in range(M):
        va = CANDS[a]
        row = div_by[a]
        row_prod = divs_of_prod[a]
        
        for y in range(M):
            if CANDS[y] % va == 0:
                div_mask[a] |= BIT[y]
            if va % CANDS[y] == 0:
                mult_mask[a] |= BIT[y]
        
        for b in range(M):
            vb = CANDS[b]
            prod_ab = va * vb
            mask_div = 0
            mask_divs = 0
            for c in range(M):
                vc = CANDS[c]
                if (vb * vc) % va == 0:
                    mask_div |= BIT[c]
                if prod_ab % vc == 0:
                    mask_divs |= BIT[c]
            row[b] = mask_div
            row_prod[b] = mask_divs
            
    return div_by, divs_of_prod, div_mask, mult_mask

_DIV_BY, _DIVS_OF_PROD, _DIV_MASK, _MULT_MASK = _precompute()
_ALL = (1 << M) - 1

def _can_add(S, x):
    if S & BIT[x]:
        return False
    
    if (_DIV_MASK[x] & S) or (_MULT_MASK[x] & S):
        return False

    dx = _DIV_BY[x]
    
    for a in range(M):
        ba = BIT[a]
        if not (S & ba):
            continue
            
        if (_DIV_BY[a][x] & S & ~ba & ~BIT[x]):
            return False
        
        if (dx[a] & S & ~ba & ~BIT[x]):
            return False
        
        if (_DIVS_OF_PROD[a][x] & S & ~ba & ~BIT[x]):
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

def _two_exchange(S, rng, trials=50):
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

def _three_exchange(S, rng, trials=30):
    """Try removing 3 and adding up to 4 to net gain +1 or more."""
    best = S
    best_sz = _popcount(S)
    for _ in range(trials):
        mems = [i for i in range(M) if S & BIT[i]]
        if len(mems) < 3:
            break
        # Remove 3 random members
        rem = rng.sample(mems, 3)
        S1 = S
        for r in rem:
            S1 &= ~BIT[r]
        
        # Try to add up to 4 new members
        S2 = S1
        added = 0
        for x in range(M):
            if added >= 4:
                break
            if (S2 & BIT[x]):
                continue
            if x in rem:
                continue
            if _can_add(S2, x):
                S2 |= BIT[x]
                added += 1
        
        S2 = _local_add(S2)
        if _popcount(S2) > best_sz:
            best = S2
            best_sz = _popcount(S2)
    return best

def construct():
    deadline = time.time() + 50.0
    best_S = 0
    best = 0

    order = list(range(M))
    S = _greedy(order)
    S = _local_add(S)
    best_S = S
    best = _popcount(S)

    for seed in range(200):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        order = list(range(M))
        rng.shuffle(order)
        S = _greedy(order)
        S = _local_add(S)
        S = _swap_improve(S, rng, rounds=8)
        S = _two_exchange(S, rng, trials=80)
        S = _three_exchange(S, rng, trials=40)
        S = _local_add(S)
        sz = _popcount(S)
        if sz > best:
            best = sz
            best_S = S

    small_members = _members(best_S)
    final_set = small_members + LARGE_PRIMES
    final_set = list(set(final_set))
    
    return final_set
