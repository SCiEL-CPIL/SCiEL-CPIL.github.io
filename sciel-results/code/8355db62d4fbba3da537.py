"""Combined program: efficient incremental greedy with multiple orderings and time guard."""
import time
import random
from itertools import combinations

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints.
    
    Only new violations can involve x as a summand or as the divisor.
    """
    n = len(selected)
    
    # Case 1: x is the divisor. Need no 3 distinct elements from selected whose sum ≡ 0 mod x.
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        return False
    
    # Case 2: x is a summand. For each a in selected, check if any 2 other elements
    # from selected\{a} plus x have sum ≡ 0 mod a.
    if n >= 2:
        for idx_a, a in enumerate(selected):
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

def _num_divisors(x):
    """Count divisors of x."""
    cnt = 0
    for d in range(1, int(x**0.5) + 1):
        if x % d == 0:
            cnt += 1
            if d != x // d:
                cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # leave 5s safety margin for 90s limit
    
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
    
    # Ordering 3: odd numbers first, then even
    if time.time() < deadline:
        odds = [x for x in range(192, 0, -1) if x % 2 == 1]
        evens = [x for x in range(192, 0, -1) if x % 2 == 0]
        order3 = odds + evens
        result3 = _greedy(order3, deadline)
        if len(result3) > len(best):
            best = result3
    
    # Ordering 4: numbers coprime to 6 first, then rest
    if time.time() < deadline:
        def coprime6(x):
            return x % 2 != 0 and x % 3 != 0
        c6 = [x for x in range(192, 0, -1) if coprime6(x)]
        rest = [x for x in range(192, 0, -1) if not coprime6(x)]
        order4 = c6 + rest
        result4 = _greedy(order4, deadline)
        if len(result4) > len(best):
            best = result4
    
    # Ordering 5: prioritize numbers with few divisors (harder to divide sums)
    if time.time() < deadline:
        order5 = sorted(range(192, 0, -1), key=lambda x: (_num_divisors(x), -x))
        result5 = _greedy(order5, deadline)
        if len(result5) > len(best):
            best = result5
    
    # Ordering 6: high numbers first, but skip very small ones that are likely to divide sums
    if time.time() < deadline:
        order6 = list(range(192, 0, -1))
        # Move small numbers (1-10) to the end since they're likely to divide many sums
        small = [x for x in order6 if x <= 10]
        large = [x for x in order6 if x > 10]
        order6 = large + small
        result6 = _greedy(order6, deadline)
        if len(result6) > len(best):
            best = result6
    
    # Ordering 7: Random restarts with different seeds
    if time.time() < deadline:
        rng = random.Random(42)
        for seed in range(10):
            if time.time() > deadline:
                break
            order7 = list(range(1, 193))
            rng.shuffle(order7)
            result7 = _greedy(order7, deadline)
            if len(result7) > len(best):
                best = result7
    
    # Ordering 8: Try to build with larger numbers preferentially, using a priority based on value
    if time.time() < deadline:
        # Sort by value descending, but with some randomization for diversity
        order8 = list(range(192, 0, -1))
        rng8 = random.Random(123)
        for i in range(len(order8)-1, 0, -1):
            j = rng8.randint(0, i)
            order8[i], order8[j] = order8[j], order8[i]
        result8 = _greedy(order8, deadline)
        if len(result8) > len(best):
            best = result8
    
    return best
