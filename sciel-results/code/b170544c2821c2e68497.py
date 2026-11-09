"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

# Precompute LCM table for all pairs in [1, 128]
_LCM_TABLE = {}
def _lcm(a, b):
    key = (min(a,b), max(a,b))
    if key not in _LCM_TABLE:
        _LCM_TABLE[key] = a * b // gcd(a, b)
    return _LCM_TABLE[key]

def _is_bad_triple(a, b, c):
    """Check if {a, b, c} forms a bad triple (equal pairwise LCMs)."""
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _conflicts_with_set(x, S):
    """Check if adding x to S creates any bad triple. Uses precomputed LCM."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        la = _lcm(x, a)
        for j in range(i + 1, n):
            b = S_list[j]
            if la == _lcm(x, b) == _lcm(a, b):
                return True
    return False

def _conflict_degree(x, S):
    """Count how many pairs in S would form bad triples with x."""
    count = 0
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        la = _lcm(x, a)
        for j in range(i + 1, n):
            b = S_list[j]
            if la == _lcm(x, b) == _lcm(a, b):
                count += 1
    return count

def _greedy_with_order(order):
    """Add elements from order to a new set if they don't conflict."""
    S = set()
    for x in order:
        if not _conflicts_with_set(x, S):
            S.add(x)
    return S

def _local_search(S, deadline):
    """
    Try to improve S by:
    1. Adding any element that doesn't conflict
    2. 1-swaps: remove one, add one
    3. 2-swaps: remove two, add two
    Returns the improved set.
    """
    S = set(S)
    full_set = set(range(1, 129))
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any element not in S
        for x in range(1, 129):
            if x in S:
                continue
            if not _conflicts_with_set(x, S):
                S.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swaps (remove one, add one)
        found_swap = False
        S_list = sorted(S)
        for a in S_list:
            S.remove(a)
            best_x = None
            best_degree = -1
            for x in range(1, 129):
                if x in S:
                    continue
                if not _conflicts_with_set(x, S):
                    deg = _conflict_degree(x, S)
                    if deg < best_degree or best_x is None:
                        best_x = x
                        best_degree = deg
            if best_x is not None:
                S.add(best_x)
                found_swap = True
                break
            else:
                S.add(a)
        
        if found_swap:
            continue
        
        # Phase 3: Try 2-swaps (remove two, add two)
        S_list = sorted(S)
        n = len(S_list)
        found_2swap = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = S_list[i], S_list[j]
                S.remove(a)
                S.remove(b)
                # Find best pair to add
                best_pair = None
                best_total_degree = -1
                for x in range(1, 129):
                    if x in S:
                        continue
                    if _conflicts_with_set(x, S):
                        continue
                    for y in range(1, 129):
                        if y in S or y == x:
                            continue
                        if not _conflicts_with_set(y, S):
                            # Check if x and y can coexist
                            if not _conflicts_with_set(x, S | {y}):
                                deg_x = _conflict_degree(x, S)
                                deg_y = _conflict_degree(y, S | {x})
                                total_deg = deg_x + deg_y
                                if total_deg < best_total_degree or best_pair is None:
                                    best_pair = (x, y)
                                    best_total_degree = total_deg
                if best_pair is not None:
                    S.add(best_pair[0])
                    S.add(best_pair[1])
                    found_2swap = True
                    break
                else:
                    S.add(a)
                    S.add(b)
                    break
            if found_2swap:
                break
        
        if not found_2swap:
            break
    
    return S

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _num_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count

def _max_prime_factor(n):
    """Return the largest prime factor of n."""
    if n < 2:
        return 1
    max_pf = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            max_pf = d
            n //= d
        d += 1
    if n > 1:
        max_pf = n
    return max_pf

def _radical(n):
    """Return the radical of n (product of distinct prime factors)."""
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            result *= d
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        result *= n
    return result

def _compute_conflict_graph():
    """Precompute which pairs of elements are in bad triples together."""
    # For each element x, compute the set of pairs (a,b) such that {x,a,b} is a bad triple
    conflict_pairs = {x: set() for x in range(1, 129)}
    for a in range(1, 129):
        for b in range(a+1, 129):
            la_b = _lcm(a, b)
            for c in range(1, 129):
                if c == a or c == b:
                    continue
                la_c = _lcm(a, c)
                lb_c = _lcm(b, c)
                if la_b == la_c == lb_c:
                    conflict_pairs[c].add((a, b))
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
    return conflict_pairs

def _conflict_degree_fast(x, S, conflict_pairs):
    """Fast conflict degree using precomputed pairs."""
    count = 0
    for (a, b) in conflict_pairs[x]:
        if a in S and b in S:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    all_elements = list(range(1, 129))
    
    # Precompute conflict pairs for fast degree calculation
    conflict_pairs = _compute_conflict_graph()
    
    # Precompute properties for all elements
    primes = [x for x in all_elements if _is_prime(x)]
    non_primes = [x for x in all_elements if not _is_prime(x)]
    div_counts = {x: _num_divisors(x) for x in all_elements}
    max_pfs = {x: _max_prime_factor(x) for x in all_elements}
    radicals = {x: _radical(x) for x in all_elements}
    
    # Compute initial conflict degree for each element (with empty set = 0)
    # We'll use this to prioritize elements with lower potential conflicts
    
    # Build several deterministic orderings
    orderings = []
    
    # 1. Primes first, then non-primes
    orderings.append(primes + non_primes)
    
    # 2. By number of divisors (fewer first)
    orderings.append(sorted(all_elements, key=lambda x: div_counts[x]))
    
    # 3. Reverse order (largest first)
    orderings.append(list(reversed(all_elements)))
    
    # 4. Forward order (smallest first) - baseline greedy
    orderings.append(all_elements)
    
    # 5. By max prime factor (larger first)
    orderings.append(sorted(all_elements, key=lambda x: max_pfs[x], reverse=True))
    
    # 6. By radical (larger first)
    orderings.append(sorted(all_elements, key=lambda x: radicals[x], reverse=True))
    
    # 7. By radical (smaller first)
    orderings.append(sorted(all_elements, key=lambda x: radicals[x]))
    
    # 8. Interleave primes and composites by value
    interleaved = []
    p_idx, c_idx = 0, 0
    while p_idx < len(primes) or c_idx < len(non_primes):
        if p_idx < len(primes):
            interleaved.append(primes[p_idx])
            p_idx += 1
        if c_idx < len(non_primes):
            interleaved.append(non_primes[c_idx])
            c_idx += 1
    orderings.append(interleaved)
    
    # 9. Sort by number of conflict pairs (fewer first)
    orderings.append(sorted(all_elements, key=lambda x: len(conflict_pairs[x])))
    
    # 10. Sort by number of conflict pairs (more first)
    orderings.append(sorted(all_elements, key=lambda x: len(conflict_pairs[x]), reverse=True))
    
    # Random orderings with different seeds
    random.seed(42)
    for _ in range(20):
        orderings.append(random.sample(all_elements, 128))
    
    best_set = set()
    best_size = 0
    
    # Run all deterministic and random greedy passes
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_with_order(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Continue with more random restarts if time permits
    while time.time() < deadline - 10:
        candidates = random.sample(all_elements, 128)
        S = _greedy_with_order(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search on the best set found
    if time.time() < deadline:
        improved = _local_search(best_set, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Final validation: ensure no bad triples exist
    S_list = sorted(best_set)
    valid = True
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            valid = False
            break
    
    if not valid:
        # Fallback: use the simplest valid set
        best_set = {1}
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
