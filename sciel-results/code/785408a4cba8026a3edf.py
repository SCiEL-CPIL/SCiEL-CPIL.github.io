"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 256}
_DEADLINE = time.time() + 75  # 75s safety margin before 90s kill

def _valid_set(s):
    """Check if set s satisfies: for all distinct a,b,c in s, a does not divide b*c."""
    lst = sorted(s)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            b = lst[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(s, x):
    """Check if adding x to set s maintains the constraint."""
    lst = sorted(s)
    n = len(lst)
    # Check all triples involving x
    for i in range(n):
        a = lst[i]
        # x and another element as b,c with a as divisor
        for j in range(n):
            if j == i:
                continue
            b = lst[j]
            if (b * x) % a == 0:
                return False
            if (x * b) % a == 0:
                return False
        # a = x, b and c from s
        for j in range(n):
            b = lst[j]
            for k in range(n):
                if k == j:
                    continue
                c = lst[k]
                if (b * c) % x == 0:
                    return False
    return True

def _greedy_add(order, n_max):
    """Greedy: try to add elements in given order."""
    s = set()
    for x in order:
        if x > n_max:
            continue
        if _can_add(s, x):
            s.add(x)
    return s

def construct():
    start = time.time()
    n_max = P['n']
    best = set()
    
    # Strategy 1: Start with primes (known valid), try to add composites
    primes = [a for a in range(2, n_max+1) if all(a%d for d in range(2, isqrt(a)+1))]
    best = set(primes)
    
    # Try to add composites in various orders
    composites = [a for a in range(2, n_max+1) if a not in set(primes)]
    
    # Order 1: composites descending
    for order in [
        sorted(composites, reverse=True),
        sorted(composites),
        sorted(composites, key=lambda x: -x.bit_length()),
        sorted(composites, key=lambda x: sum(1 for d in range(2, isqrt(x)+1) if x%d==0), reverse=True),
    ]:
        if time.time() - start > _DEADLINE - start:
            break
        s = set(primes)
        for x in order:
            if time.time() - start > _DEADLINE - start:
                break
            if _can_add(s, x):
                s.add(x)
        if len(s) > len(best):
            best = s
    
    # Strategy 2: Pure greedy from scratch with different orderings
    all_nums = list(range(2, n_max+1))
    for order in [
        sorted(all_nums, reverse=True),
        sorted(all_nums, key=lambda x: -x.bit_length()),
        sorted(all_nums, key=lambda x: x),
    ]:
        if time.time() - start > _DEADLINE - start:
            break
        s = _greedy_add(order, n_max)
        if len(s) > len(best):
            best = s
    
    # Strategy 3: Start with large numbers only
    large = list(range(n_max//2 + 1, n_max+1))
    for order in [sorted(large, reverse=True), sorted(large)]:
        if time.time() - start > _DEADLINE - start:
            break
        s = _greedy_add(order, n_max)
        if len(s) > len(best):
            best = s
    
    # Local search: try to swap out one element for another
    if time.time() - start < _DEADLINE - start:
        current = list(best)
        improved = True
        while improved and time.time() - start < _DEADLINE - start:
            improved = False
            for i in range(len(current)):
                if time.time() - start > _DEADLINE - start:
                    break
                trial = set(current)
                trial.remove(current[i])
                # Try to add any number not in trial
                for x in range(2, n_max+1):
                    if x in trial:
                        continue
                    if _can_add(trial, x):
                        trial.add(x)
                        if len(trial) > len(current):
                            current = list(trial)
                            best = set(trial)
                            improved = True
                        break
    
    result = sorted(best)
    # Ensure types are correct
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
