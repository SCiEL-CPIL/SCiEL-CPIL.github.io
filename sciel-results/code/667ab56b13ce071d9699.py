"""Construct a set of integers (<=192) maximizing cardinality such that no element
divides the sum of any 3 distinct other elements."""
import time
from itertools import combinations

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints."""
    # For each existing element a in selected, check no 3-combination of (selected \ {a})
    # has sum divisible by a. But we only need to check combos involving x since
    # existing elements were already validated among themselves.
    # Actually we need to check: for each a in selected ∪ {x}, no 3-distinct-others sum ≡ 0 mod a.
    # New violations can only involve x as a summand or as the divisor.
    
    # Case 1: x is the divisor. Need no 3 distinct elements from selected whose sum ≡ 0 mod x.
    n = len(selected)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if (selected[i] + selected[j] + selected[k]) % x == 0:
                    return False
    
    # Case 2: x is a summand. For each a in selected, check if any 2 other elements
    # from selected\{a} plus x have sum ≡ 0 mod a.
    for idx_a, a in enumerate(selected):
        # Need pairs (i,j) from selected excluding a such that (sel[i]+sel[j]+x) % a == 0
        for i in range(n):
            if i == idx_a:
                continue
            for j in range(i+1, n):
                if j == idx_a:
                    continue
                if (selected[i] + selected[j] + x) % a == 0:
                    return False
    
    return True

def _greedy(order, deadline):
    """Greedy construction following given order, respecting deadline."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # leave 10s safety margin
    
    # Try multiple orderings
    best = []
    
    # Ordering 1: descending (original)
    order1 = list(range(192, 0, -1))
    result1 = _greedy(order1, deadline)
    best = result1
    
    # Ordering 2: ascending
    if time.time() < deadline:
        order2 = list(range(1, 193))
        result2 = _greedy(order2, deadline)
        if len(result2) > len(best):
            best = result2
    
    # Ordering 3: larger numbers first but skip small ones initially
    if time.time() < deadline:
        order3 = list(range(192, 96, -1)) + list(range(96, 0, -1))
        result3 = _greedy(order3, deadline)
        if len(result3) > len(best):
            best = result3
    
    # Ordering 4: only odd numbers first, then even
    if time.time() < deadline:
        odds = [x for x in range(192, 0, -1) if x % 2 == 1]
        evens = [x for x in range(192, 0, -1) if x % 2 == 0]
        order4 = odds + evens
        result4 = _greedy(order4, deadline)
        if len(result4) > len(best):
            best = result4
    
    # Ordering 5: numbers coprime to 6 first, then rest
    if time.time() < deadline:
        def coprime6(x):
            return x % 2 != 0 and x % 3 != 0
        c6 = [x for x in range(192, 0, -1) if coprime6(x)]
        rest = [x for x in range(192, 0, -1) if not coprime6(x)]
        order5 = c6 + rest
        result5 = _greedy(order5, deadline)
        if len(result5) > len(best):
            best = result5
    
    # Ordering 6: high numbers, prioritize those with few divisors
    if time.time() < deadline:
        def num_divisors(x):
            cnt = 0
            for d in range(1, int(x**0.5)+1):
                if x % d == 0:
                    cnt += 1
                    if d != x//d:
                        cnt += 1
            return cnt
        order6 = sorted(range(192, 0, -1), key=lambda x: (num_divisors(x), -x))
        result6 = _greedy(order6, deadline)
        if len(result6) > len(best):
            best = result6
    
    return best
# EVOLVE-BLOCK-END
