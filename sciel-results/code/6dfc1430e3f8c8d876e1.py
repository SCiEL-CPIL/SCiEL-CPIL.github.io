"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(max_n):
    """Precompute LCM table for all pairs in [1, max_n]."""
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    Returns a list of tuples (a,b,c).
    """
    n = max_n
    lcm_table = _precompute_lcms(n)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    return bad_triples, lcm_table

def _greedy_with_conflicts(order, conflict_pairs, max_n):
    """
    Greedy: iterate through order, add x if it doesn't form a bad triple
    with any two already-selected elements.
    conflict_pairs[x] = list of (y, z) such that {x,y,z} is a bad triple.
    """
    selected = []
    selected_set = set()
    for x in order:
        ok = True
        # Check all pairs in selected
        # For speed, we can check: for each y in selected, is (x,y) in some bad triple with z in selected?
        # Actually, let's just check: for each pair (y,z) in selected, is {x,y,z} a bad triple?
        # That's O(|selected|^2) per x.
        # Better: precompute for each x, the set of pairs (y,z) that form a bad triple with x.
        # Then check if any such pair is fully in selected.
        for y, z in conflict_pairs.get(x, []):
            if y in selected_set and z in selected_set:
                ok = False
                break
        if ok:
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, conflict_pairs, numbers, deadline):
    """
    Try to add any element that can be added, then try swaps.
    """
    selected_set = set(selected)
    best = list(selected)
    best_len = len(best)
    
    # Phase 1: Add any element that fits
    changed = True
    while changed:
        changed = False
        for x in numbers:
            if time.time() > deadline:
                return best
            if x in selected_set:
                continue
            ok = True
            for y, z in conflict_pairs.get(x, []):
                if y in selected_set and z in selected_set:
                    ok = False
                    break
            if ok:
                best.append(x)
                selected_set.add(x)
                best_len = len(best)
                changed = True
    
    # Phase 2: Try swaps to increase size
    for i in range(len(best)):
        if time.time() > deadline:
            break
        removed = best[i]
        test_set = selected_set - {removed}
        for x in numbers:
            if x in test_set:
                continue
            # Check if x can be added to test_set
            ok = True
            for y, z in conflict_pairs.get(x, []):
                if y in test_set and z in test_set:
                    ok = False
                    break
            if ok:
                # We found a swap that maintains size, but we want to increase size.
                # A swap doesn't increase size. So this phase is for maintaining size while enabling future additions.
                # Actually, let's skip swaps for now and focus on additions.
                pass
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    max_n = 128
    numbers = list(range(1, max_n + 1))
    
    # Precompute LCM table and bad triples
    bad_triples, lcm_table = _precompute_bad_triples(max_n)
    
    # Build conflict_pairs: for each x, list of (y,z) pairs such that {x,y,z} is bad
    conflict_pairs = {x: [] for x in numbers}
    for a, b, c in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    # Strategy 1: Primes and Prime Powers first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    order1 = primes + prime_powers + others
    result1 = _greedy_with_conflicts(order1, conflict_pairs, max_n)
    
    # Order 2: Standard 1..128
    order2 = numbers
    result2 = _greedy_with_conflicts(order2, conflict_pairs, max_n)
    
    # Order 3: Sorted by number of divisors (ascending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order3 = sorted(numbers, key=num_divisors)
    result3 = _greedy_with_conflicts(order3, conflict_pairs, max_n)
    
    best_result = max([result1, result2, result3], key=len)
    
    # Try many random shuffles if time permits
    random.seed(42)
    for i in range(2000):
        if time.time() > deadline:
            break
        shuffled = numbers[:]
        random.shuffle(shuffled)
        res = _greedy_with_conflicts(shuffled, conflict_pairs, max_n)
        if len(res) > len(best_result):
            best_result = res
    
    # Local search improvement on the best result
    if time.time() < deadline - 5:
        improved = _local_search(best_result, conflict_pairs, numbers, deadline)
        if len(improved) > len(best_result):
            best_result = improved
    
    return best_result
# EVOLVE-BLOCK-END
