import time
import random

N = 128
DEADLINE = 55.0

def _check_add(mask, x):
    """Check if adding x to mask keeps the set valid."""
    if x > N or x < 1:
        return False
    if (mask >> x) & 1:
        return True  # already in set
    # x is not in mask. Check all pairs (a,b) in mask with a<b.
    # If a+b == x, then we need to check that no c in mask (c!=a,b) has a+c in mask and b+c in mask.
    # Also, if x+a in mask and x+b in mask and a+b in mask, that creates a forbidden triple.
    # Actually the constraint is: no distinct a,b,c in S with a+b, a+c, b+c all in S.
    # When we add x, new forbidden triples must involve x.
    # Case 1: x is one of a,b,c. WLOG x=a. Then we need b,c in S (old mask), and b+c, x+c, x+b in S.
    #   But x+b and x+c must be in S. Since x is new, x+b and x+c must be in old mask.
    #   And b+c must be in old mask.
    # Case 2: x is one of the sums. WLOG x=a+b. Then a,b,c in S (old mask), and a+c, b+c in S (old mask).
    #   So we need a,b,c distinct in old mask with a+b==x, a+c in old mask, b+c in old mask.
    
    # For efficiency, iterate over pairs in mask.
    elems = [i for i in range(1, N + 1) if (mask >> i) & 1]
    m = len(elems)
    
    # Precompute a set for O(1) lookup
    def has(y):
        return y <= N and ((mask >> y) & 1) != 0
    
    # Case 2: x = a+b for some a,b in mask. Check if there exists c in mask, c!=a,b, with a+c in mask and b+c in mask.
    for i in range(m):
        a = elems[i]
        if a >= x:
            break
        b = x - a
        if b <= a or b > N:
            continue
        if not has(b):
            continue
        # a,b in mask, a+b==x. Now find c in mask, c!=a,b, with a+c in mask and b+c in mask.
        for k in range(m):
            c = elems[k]
            if c == a or c == b:
                continue
            if has(a + c) and has(b + c):
                return False
    
    # Case 1: x is one of the elements. WLOG x=a. Need b,c in mask, b!=c, b!=x, c!=x,
    #   with b+c in mask, x+c in mask, x+b in mask.
    #   So b+c in mask, x+b in mask, x+c in mask.
    for i in range(m):
        b = elems[i]
        if b == x:
            continue
        xb = x + b
        if xb > N or not has(xb):
            continue
        for j in range(m):
            c = elems[j]
            if c == x or c == b:
                continue
            xc = x + c
            if xc > N or not has(xc):
                continue
            bc = b + c
            if bc > N or not has(bc):
                continue
            return False
    
    return True

def _popcount(x):
    return bin(x).count('1')

def _greedy_start():
    mask = 0
    for i in range(N // 2 + 1, N + 1):
        mask |= (1 << i)
    for i in range(1, N // 2 + 1):
        if (mask >> i) & 1:
            continue
        if _check_add(mask, i):
            mask |= (1 << i)
    return mask

def _local_search(mask, time_limit, rng):
    best = mask
    best_size = _popcount(best)
    while time.time() < time_limit:
        elems = [i for i in range(1, N + 1) if (best >> i) & 1]
        if not elems:
            break
        remove = rng.choice(elems)
        trial = best & ~(1 << remove)
        candidates = [i for i in range(1, N + 1) if not (trial >> i) & 1]
        rng.shuffle(candidates)
        cur = trial
        for i in candidates:
            if _check_add(cur, i):
                cur |= (1 << i)
        if _popcount(cur) > best_size:
            best = cur
            best_size = _popcount(cur)
        for i in range(1, N + 1):
            if (best >> i) & 1:
                continue
            if _check_add(best, i):
                best |= (1 << i)
                best_size += 1
                break
    return best

def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)
    best = _greedy_start()
    best_size = _popcount(best)
    while time.time() < deadline - 2.0:
        mask = _local_search(best, deadline - 2.0, rng)
        if _popcount(mask) > best_size:
            best = mask
            best_size = _popcount(mask)
        else:
            if rng.random() < 0.3:
                mask = _greedy_start()
                if _popcount(mask) > best_size:
                    best = mask
                    best_size = _popcount(mask)
    result = [i for i in range(1, N + 1) if (best >> i) & 1]
    return list(result)
