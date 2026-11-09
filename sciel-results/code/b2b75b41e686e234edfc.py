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

# Precompute conflicts: _CONFLICTS[x] = set of (i,j) with i<j, i!=x, j!=x
# such that x conflicts with pair (i,j), i.e., (i*j)%x==0 or (x*j)%i==0 or (x*i)%j==0
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

def _pair_key(a, b):
    """Return canonical (min, max) pair."""
    if a > b:
        a, b = b, a
    return (a, b)

def _can_add_fast(S_pairs, x):
    """Check if x can be added given the set of all pairs in S (as a set of (i,j) tuples)."""
    # Check intersection between _CONFLICTS[x] and S_pairs
    for pair in _CONFLICTS[x]:
        if pair in S_pairs:
            return False
    return True

def _can_add_slow(S_list, x):
    """Fallback check (slower, for verification)."""
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                return False
    return True

def _build_pairs(S_list):
    """Build set of all pairs from S_list."""
    pairs = set()
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.add(_pair_key(S_list[i], S_list[j]))
    return pairs

def _num_distinct_prime_factors(n):
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

def _conflict_degree(x, S_set):
    """Count how many pairs in S_set conflict with x."""
    count = 0
    for pair in _CONFLICTS[x]:
        if pair in S_set:
            count += 1
    return count

def _greedy_build(ordering, deadline):
    """Build initial set using greedy approach with given ordering."""
    S = []
    S_set = set()
    S_pairs = set()
    candidates = list(ordering)
    
    while time.time() < deadline and candidates:
        valid = []
        for x in candidates:
            if x not in S_set and _can_add_fast(S_pairs, x):
                valid.append(x)
        
        if not valid:
            break
        
        best_x = valid[0]
        S.append(best_x)
        S_set.add(best_x)
        candidates.remove(best_x)
        # Update pairs
        for a in S[:-1]:
            S_pairs.add(_pair_key(a, best_x))
    
    return S, S_set, S_pairs

def _try_add_greedy(S_list, S_set, S_pairs, deadline, max_adds=None):
    """Try to greedily add elements to S_list. Returns (new_list, new_set, new_pairs, added_count)."""
    S_after = list(S_list)
    S_after_set = set(S_after)
    S_after_pairs = set(S_pairs)
    added = 0
    if max_adds is None:
        max_adds = _N
    for c in range(2, _N + 1):
        if time.time() > deadline:
            break
        if added >= max_adds:
            break
        if c in S_after_set:
            continue
        if _can_add_fast(S_after_pairs, c):
            S_after.append(c)
            S_after_set.add(c)
            for a in S_after[:-1]:
                S_after_pairs.add(_pair_key(a, c))
            added += 1
    return S_after, S_after_set, S_after_pairs, added

def _local_search(S, S_set, S_pairs, deadline):
    """Local search: try adding elements, then swaps."""
    S = list(S)
    S_set = set(S)
    S_pairs = set(S_pairs)
    
    # Phase 1: Try to add any element that can be added
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in range(2, _N + 1):
            if a in S_set:
                continue
            if _can_add_fast(S_pairs, a):
                S.append(a)
                S_set.add(a)
                for b in S[:-1]:
                    S_pairs.add(_pair_key(a, b))
                improved = True
    
    # Phase 2: 1-for-2 swap local search (remove one, try to add many)
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
            S_temp_pairs = _build_pairs(S_temp)
            
            # Try to add elements to S_temp
            S_after, S_after_set, S_after_pairs, _ = _try_add_greedy(S_temp, S_temp_set, S_temp_pairs, deadline)
            
            if len(S_after) > current_size:
                S = S_after
                S_set = S_after_set
                S_pairs = S_after_pairs
                improved = True
                break
    
    # Phase 3: 2-for-2 swap (remove two, try to add many)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(S)
        n = len(S)
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                # Remove S[i] and S[j]
                S_temp = [x for k, x in enumerate(S) if k != i and k != j]
                S_temp_set = set(S_temp)
                S_temp_pairs = _build_pairs(S_temp)
                
                # Try to add elements to S_temp
                S_after, S_after_set, S_after_pairs, _ = _try_add_greedy(S_temp, S_temp_set, S_temp_pairs, deadline)
                
                if len(S_after) > current_size:
                    S = S_after
                    S_set = S_after_set
                    S_pairs = S_after_pairs
                    improved = True
                    break
            
            if improved:
                break
    
    # Phase 4: 1-for-1 swap with follow-up additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(S)
        
        for i, a in enumerate(list(S)):
            if time.time() > deadline:
                break
            
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            S_temp_pairs = _build_pairs(S_temp)
            
            for b in range(2, _N + 1):
                if b in S_temp_set:
                    continue
                if _can_add_fast(S_temp_pairs, b):
                    S_after = list(S_temp) + [b]
                    S_after_set = set(S_after)
                    S_after_pairs = set(S_temp_pairs)
                    for c in S_after[:-1]:
                        S_after_pairs.add(_pair_key(b, c))
                    
                    # Try to add more
                    S_after, S_after_set, S_after_pairs, _ = _try_add_greedy(S_after, S_after_set, S_after_pairs, deadline)
                    
                    if len(S_after) > current_size:
                        S = S_after
                        S_set = S_after_set
                        S_pairs = S_after_pairs
                        improved = True
                        break
            
            if improved:
                break
    
    return S, S_set, S_pairs

def _diversified_search(deadline):
    """Try multiple diversified greedy orderings and keep the best result."""
    best_S = []
    best_S_set = set()
    best_S_pairs = set()
    
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
    
    # Ordering 7: Sort by conflict degree (fewer conflicts first) - approximate
    # Use number of conflicts as a proxy: numbers with fewer total conflicts are "safer"
    conflict_counts = []
    for x in all_numbers:
        conflict_counts.append((len(_CONFLICTS[x]), x))
    conflict_counts.sort()
    orderings.append([x for _, x in conflict_counts])
    
    # Ordering 8: Large primes first, then large composites
    large_primes = [a for a in primes if a > 16]
    small_primes = [a for a in primes if a <= 16]
    large_composites = [a for a in composites if a > 16]
    small_composites = [a for a in composites if a <= 16]
    orderings.append(large_primes + large_composites + small_primes + small_composites)
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        S, S_set, S_pairs = _greedy_build(ordering, deadline)
        S, S_set, S_pairs = _local_search(S, S_set, S_pairs, deadline)
        
        if len(S) > len(best_S):
            best_S = S
            best_S_set = S_set
            best_S_pairs = S_pairs
    
    return best_S, best_S_set, best_S_pairs

def _verify(S):
    """Verify the set is valid."""
    n = len(S)
    S_set = set(S)
    if len(S_set) != n:
        return False
    for x in S:
        if x < 2 or x > 256:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S[i], S[j], S[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Use diversified search
    S, S_set, S_pairs = _diversified_search(deadline)
    
    # Final local search polish
    S, S_set, S_pairs = _local_search(S, S_set, S_pairs, deadline)
    
    # Verify validity
    if not _verify(S):
        # Fallback: try a simple greedy
        S, S_set, S_pairs = _greedy_build(list(range(2, _N + 1, -1)), deadline)
        S, S_set, S_pairs = _local_search(S, S_set, S_pairs, deadline)
    
    # Return as list of Python ints
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
