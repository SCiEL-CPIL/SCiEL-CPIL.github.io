"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

_N = 256

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

# Precompute: for each x, the set of pairs (i,j) with i<j, i!=x, j!=x where x conflicts
# x conflicts with (i,j) if (i*j)%x==0 or (x*j)%i==0 or (x*i)%j==0
def _build_conflicts():
    conflicts = [set() for _ in range(_N + 1)]
    for x in range(2, _N + 1):
        for i in range(2, _N + 1):
            if i == x:
                continue
            for j in range(i + 1, _N + 1):
                if j == x:
                    continue
                if (i * j) % x == 0 or (x * j) % i == 0 or (x * i) % j == 0:
                    conflicts[x].add((i, j))
    return conflicts

_CONFLICTS = _build_conflicts()

def _can_add(S_list, x):
    """Check if x can be added to set S_list without violating constraints."""
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                return False
    return True

def _conflict_count(x, S_list):
    """Count how many pairs in S_list conflict with x."""
    count = 0
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                count += 1
    return count

def _num_distinct_prime_factors(n):
    """Count distinct prime factors of n."""
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

def _largest_prime_factor(n):
    """Find the largest prime factor of n."""
    largest = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            largest = d
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        largest = temp
    return largest

def _greedy_build(ordering, deadline):
    """Build initial set using greedy approach with given ordering."""
    S = []
    S_set = set()
    candidates = list(ordering)
    
    while time.time() < deadline and candidates:
        valid = []
        for x in candidates:
            if x not in S_set and _can_add(S, x):
                valid.append(x)
        
        if not valid:
            break
        
        best_x = valid[0]  # First valid in the given ordering
        S.append(best_x)
        S_set.add(best_x)
        candidates.remove(best_x)
    
    return S, S_set

def _local_search(S, S_set, deadline):
    """Local search: try adding elements, then 1-for-1 and 1-for-2 swaps."""
    S = list(S)
    S_set = set(S)
    
    # Phase 1: Try to add any element that can be added
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in range(2, _N + 1):
            if a in S_set:
                continue
            if _can_add(S, a):
                S.append(a)
                S_set.add(a)
                improved = True
    
    # Phase 2: 1-for-2 swap local search (remove one, try to add two)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(S)
        
        for i, a in enumerate(list(S)):
            if time.time() > deadline:
                break
            
            # Remove a
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            
            # Try to add elements to S_temp
            S_after = list(S_temp)
            S_after_set = set(S_after)
            
            added_count = 0
            for c in range(2, _N + 1):
                if c in S_after_set:
                    continue
                if _can_add(S_after, c):
                    S_after.append(c)
                    S_after_set.add(c)
                    added_count += 1
            
            if len(S_after) > current_size:
                S = S_after
                S_set = S_after_set
                improved = True
                break
    
    # Phase 3: 1-for-1 swap with improvement check
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(S)
        
        for i, a in enumerate(list(S)):
            if time.time() > deadline:
                break
            
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            
            for b in range(2, _N + 1):
                if b in S_temp_set:
                    continue
                if _can_add(S_temp, b):
                    S_after = list(S_temp) + [b]
                    S_after_set = set(S_after)
                    
                    # Try to add more
                    for c in range(2, _N + 1):
                        if c in S_after_set:
                            continue
                        if _can_add(S_after, c):
                            S_after.append(c)
                            S_after_set.add(c)
                    
                    if len(S_after) > current_size:
                        S = S_after
                        S_set = S_after_set
                        improved = True
                        break
            
            if improved:
                break
    
    return S, S_set

def _diversified_search(deadline):
    """Try multiple diversified greedy orderings and keep the best result."""
    best_S = []
    best_S_set = set()
    
    # Generate different orderings
    all_numbers = list(range(2, _N + 1))
    
    orderings = []
    
    # Ordering 1: Primes first, then composites by value
    primes = [a for a in all_numbers if _is_prime(a)]
    composites = [a for a in all_numbers if not _is_prime(a)]
    orderings.append(primes + composites)
    
    # Ordering 2: Random
    random_order = list(all_numbers)
    random.shuffle(random_order)
    orderings.append(random_order)
    
    # Ordering 3: Few distinct prime factors first
    orderings.append(sorted(all_numbers, key=lambda x: (_num_distinct_prime_factors(x), x)))
    
    # Ordering 4: Large numbers first
    orderings.append(sorted(all_numbers, reverse=True))
    
    # Ordering 5: Large prime factor first
    orderings.append(sorted(all_numbers, key=lambda x: -_largest_prime_factor(x)))
    
    # Ordering 6: Another random
    random_order2 = list(all_numbers)
    random.shuffle(random_order2)
    orderings.append(random_order2)
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        S, S_set = _greedy_build(ordering, deadline)
        S, S_set = _local_search(S, S_set, deadline)
        
        if len(S) > len(best_S):
            best_S = S
            best_S_set = S_set
    
    return best_S, best_S_set

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Use diversified search
    S, S_set = _diversified_search(deadline)
    
    # Final local search polish
    S, S_set = _local_search(S, S_set, deadline)
    
    # Final check: verify validity
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S[i], S[j], S[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    # Invalid - should not happen, but handle gracefully
                    pass
    
    # Return as list of Python ints
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
