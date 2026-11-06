"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _is_valid_set(S):
    """Check if set S has no three distinct elements with equal pairwise GCDs."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = S_list[i], S_list[j], S_list[k]
                g1 = gcd(a, b)
                g2 = gcd(a, c)
                g3 = gcd(b, c)
                if g1 == g2 == g3:
                    return False
    return True

def _greedy_build(order, deadline):
    """Greedy: add elements in given order if valid."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        # Check if adding x creates a bad triple
        ok = True
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                a, b = selected[i], selected[j]
                g_ab = gcd(a, b)
                g_ax = gcd(a, x)
                g_bx = gcd(b, x)
                if g_ab == g_ax == g_bx:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
    return selected

def construct():
    start = time.time()
    deadline = start + 80.0  # 10s safety margin
    
    all_nums = list(range(1, 129))
    
    best_set = []
    
    # Strategy 1: All prime powers (p^k <= 128)
    prime_powers = set()
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]:
        pk = p
        while pk <= 128:
            prime_powers.add(pk)
            pk *= p
    # Also add 1
    prime_powers.add(1)
    pp_list = sorted(prime_powers)
    s1 = _greedy_build(pp_list, deadline)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: All even numbers (multiples of 2)
    evens = list(range(2, 129, 2))
    s2 = _greedy_build(evens, deadline)
    if len(s2) > len(best_set):
        best_set = s2
    
    # Strategy 3: All multiples of 3
    mult3 = list(range(3, 129, 3))
    s3 = _greedy_build(mult3, deadline)
    if len(s3) > len(best_set):
        best_set = s3
    
    # Strategy 4: Numbers with exactly one prime factor (prime powers) + 1
    # Try different orderings
    # Order by descending number of prime factors (fewer = safer)
    from collections import Counter
    def num_distinct_primes(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    # Strategy 5: Greedy with ordering by descending value
    desc_order = list(range(128, 0, -1))
    s5 = _greedy_build(desc_order, deadline)
    if len(s5) > len(best_set):
        best_set = s5
    
    # Strategy 6: Greedy with ordering by ascending value
    asc_order = list(range(1, 129))
    s6 = _greedy_build(asc_order, deadline)
    if len(s6) > len(best_set):
        best_set = s6
    
    # Strategy 7: Try to build a large set using local search / hill climbing
    # Start from best greedy, try to add more elements
    if time.time() < deadline - 5.0:
        current = list(best_set)
        current_set = set(current)
        candidates = [x for x in range(1, 129) if x not in current_set]
        # Try adding each candidate if it doesn't break validity
        for x in candidates:
            if time.time() > deadline - 2.0:
                break
            # Check if x can be added
            ok = True
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    a, b = current[i], current[j]
                    g_ab = gcd(a, b)
                    g_ax = gcd(a, x)
                    g_bx = gcd(b, x)
                    if g_ab == g_ax == g_bx:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                current.append(x)
                current_set.add(x)
        if len(current) > len(best_set):
            best_set = current
    
    # Strategy 8: Try multiple random orderings
    import random
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 2.0:
            break
        rand_order = list(range(1, 129))
        random.shuffle(rand_order)
        s_rand = _greedy_build(rand_order, deadline)
        if len(s_rand) > len(best_set):
            best_set = s_rand
    
    # Final verification
    if best_set:
        if not _is_valid_set(best_set):
            # Fallback: use a known valid small set
            best_set = [1]
    else:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
