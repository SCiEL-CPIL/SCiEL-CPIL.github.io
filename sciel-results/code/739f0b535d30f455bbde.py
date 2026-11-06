"""Rewritten program: greedy + local search for equal-lcm-triple-free subsets of [1,128]."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 80.0  # 80s safety margin under 90s kill

def _build_lcm_table(n=128):
    """Precompute lcm table for 1..n."""
    from math import gcd
    table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            table[i][j] = table[j][i] = i * j // gcd(i, j)
    return table

def _is_bad_triple(a, b, c, L):
    """Check if {a,b,c} form a bad triple (three pairwise LCMS equal)."""
    return L[a][b] == L[a][c] == L[b][c]

def _can_add(x, selected_set, L):
    """Check if x can be added to selected_set without forming a bad triple."""
    lst = list(selected_set)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if _is_bad_triple(lst[i], lst[j], x, L):
                return False
    return True

def _validate(S, L):
    """Verify S has no bad triple. Returns True if valid."""
    lst = sorted(S)
    for a, b, c in combinations(lst, 3):
        if _is_bad_triple(a, b, c, L):
            return False
    return True

def _greedy(L, order):
    """Greedy: add elements in given order if they don't create a bad triple."""
    selected = set()
    for x in order:
        if x not in selected and _can_add(x, selected, L):
            selected.add(x)
    return selected

def _local_search(S, L, max_rounds=200):
    """Local search: try swapping a selected element with an unselected one."""
    S = set(S)
    unselected = set(range(1, 129)) - S
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < _DEADLINE:
        improved = False
        rounds += 1
        # Try removing one element and adding two (or one) unselected
        for rem in sorted(S):
            if time.time() > _DEADLINE:
                break
            S_without = S - {rem}
            # Try adding one unselected element
            for add in sorted(unselected):
                if add in S:
                    continue
                S_candidate = S_without | {add}
                if len(S_candidate) == len(S) and _validate(S_candidate, L):
                    # same size but different composition; accept if it opens more room
                    # Actually for cardinality we want larger; try to add more
                    pass
            # Try adding two unselected elements (net +1)
            for add1, add2 in combinations(sorted(unselected), 2):
                if time.time() > _DEADLINE:
                    break
                S_candidate = S_without | {add1, add2}
                if len(S_candidate) == len(S) + 1 and _validate(S_candidate, L):
                    S = S_candidate
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    L = _build_lcm_table(128)
    
    best = set()
    
    # Deterministic greedy: ascending order
    order_asc = list(range(1, 129))
    best = _greedy(L, order_asc)
    
    # Deterministic greedy: descending order
    order_desc = list(range(128, 0, -1))
    s = _greedy(L, order_desc)
    if len(s) > len(best):
        best = s
    
    # Greedy by various heuristic orderings
    # Order by number of divisors (fewer divisors = more "prime-like", safer)
    from math import gcd
    def num_divisors(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c
    
    # Order by largest prime factor (heuristic)
    def largest_prime_factor(n):
        f = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                f = d
                n //= d
            d += 1
        if n > 1:
            f = max(f, n)
        return f
    
    order_lpf = sorted(range(1, 129), key=lambda x: -largest_prime_factor(x))
    s = _greedy(L, order_lpf)
    if len(s) > len(best):
        best = s
    
    # Randomized greedy passes
    import random
    rng = random.Random(42)
    
    for _ in range(50):
        if time.time() > _DEADLINE - 5.0:
            break
        order = list(range(1, 129))
        rng.shuffle(order)
        s = _greedy(L, order)
        if len(s) > len(best):
            best = s
    
    # Local search on best
    if time.time() < _DEADLINE - 5.0:
        s = _local_search(best, L, max_rounds=100)
        if len(s) > len(best):
            best = s
    
    # More randomized + local search
    for _ in range(30):
        if time.time() > _DEADLINE - 10.0:
            break
        order = list(range(1, 129))
        rng.shuffle(order)
        s = _greedy(L, order)
        if time.time() < _DEADLINE - 5.0:
            s = _local_search(s, L, max_rounds=50)
        if len(s) > len(best):
            best = s
    
    # Final local search on best
    if time.time() < _DEADLINE - 2.0:
        s = _local_search(best, L, max_rounds=200)
        if len(s) > len(best):
            best = s
    
    # Ensure nonempty
    if not best:
        best = {1}
    
    return sorted(best)
# EVOLVE-BLOCK-END
