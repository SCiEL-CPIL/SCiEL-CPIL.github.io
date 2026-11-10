"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def _can_add(S, S_set, x):
    """Check if x can be added to S without creating a forbidden triple."""
    if x in S_set:
        return False
    
    n = len(S)
    
    # Case 1: x is one of the distinct members (a, b, c). Let c = x.
    # We need distinct a, b in S such that:
    # 1. a + x is in S (since a+x > x, it must be in old S)
    # 2. b + x is in S (since b+x > x, it must be in old S)
    # 3. a + b is in S OR a + b == x
    for i in range(n):
        a = S[i]
        if (a + x) not in S_set:
            continue
        for j in range(i + 1, n):
            b = S[j]
            if (b + x) not in S_set:
                continue
            # a, b, x are distinct (a!=b, a!=x, b!=x)
            if (a + b) in S_set or (a + b) == x:
                return False

    # Case 2: x is one of the sums. Let a + b = x.
    # We need distinct a, b, c in S such that:
    # 1. a + b == x
    # 2. a + c is in S
    # 3. b + c is in S
    # Note: a, b, c must be distinct.
    for a in S:
        b = x - a
        if b in S_set and a != b:
            # a and b are distinct elements in S summing to x.
            # Now find c in S, c != a, c != b, such that a+c in S and b+c in S.
            for c in S:
                if c == a or c == b:
                    continue
                if (a + c) in S_set and (b + c) in S_set:
                    return False
                    
    return True

def _greedy_from_order(order):
    S = []
    S_set = set()
    for x in order:
        if _can_add(S, S_set, x):
            S.append(x)
            S_set.add(x)
    return S

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    rng = random.Random(42)
    
    # Strategy 1: Descending order (large numbers first)
    # Large numbers are less likely to form sums within the range 1..N
    order1 = list(range(n, 0, -1))
    best = _greedy_from_order(order1)
    best_size = len(best)
    
    # Strategy 2: Ascending order
    order2 = list(range(1, n + 1))
    S2 = _greedy_from_order(order2)
    if len(S2) > best_size:
        best = S2
        best_size = len(S2)

    # Strategy 3: Random restarts
    for _ in range(500):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        S = _greedy_from_order(order)
        if len(S) > best_size:
            best = S
            best_size = len(S)
            
    # Final pass: try to add any missing element
    best_set = set(best)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x not in best_set and _can_add(best, best_set, x):
            best.append(x)
            best_set.add(x)
    
    return best

# EVOLVE-BLOCK-END
