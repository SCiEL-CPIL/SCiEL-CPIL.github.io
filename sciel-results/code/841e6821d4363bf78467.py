"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt
from itertools import combinations

def _sieve(n):
    """Return list of all primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(2, n+1) if is_prime[i]]

def _is_valid_set(s):
    """Check if for every three distinct a,b,c in s, a does not divide b*c."""
    s_list = list(s)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if j == i: continue
            b = s_list[j]
            for k in range(n):
                if k == i or k == j: continue
                c = s_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy_remove_smallest(full_set):
    """
    Greedy approach: start with full_set, repeatedly find a violating triple
    and remove the smallest element from it until valid.
    """
    s = set(full_set)
    changed = True
    while changed:
        changed = False
        s_list = sorted(s)
        n = len(s_list)
        found = False
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i: continue
                b = s_list[j]
                for k in range(n):
                    if k == i or k == j: continue
                    c = s_list[k]
                    if (b * c) % a == 0:
                        # Remove the smallest of a, b, c
                        min_elem = min(a, b, c)
                        s.discard(min_elem)
                        found = True
                        changed = True
                        break
                if found:
                    break
            if found:
                break
    return s

def _greedy_add_largest(prime_set, n=256):
    """
    Start with primes, try to add composites from largest to smallest
    if they don't create a violation.
    """
    s = set(prime_set)
    for candidate in range(n, 1, -1):
        if candidate in s:
            continue
        s.add(candidate)
        if not _is_valid_set(s):
            s.discard(candidate)
    return s

def _local_search(s, n=256, iterations=10000):
    """
    Local search: try swaps (remove one, add another) to improve size.
    """
    current = set(s)
    current_size = len(current)
    candidates = [i for i in range(2, n+1) if i not in current]
    
    for _ in range(iterations):
        improved = False
        for out_elem in list(current):
            for in_elem in candidates:
                if in_elem == out_elem:
                    continue
                new_set = current - {out_elem} | {in_elem}
                if len(new_set) != current_size:
                    continue
                if _is_valid_set(new_set):
                    # Check if it's actually better (same size but maybe allows future improvements)
                    # For simplicity, just take the first valid swap
                    current = new_set
                    candidates = [i for i in range(2, n+1) if i not in current]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return current

def construct():
    n = 256
    primes = _sieve(n)
    prime_set = set(primes)
    
    # Strategy 1: Greedy removal from full set
    s1 = _greedy_remove_smallest(set(range(2, n+1)))
    
    # Strategy 2: Greedy add from primes
    s2 = _greedy_add_largest(prime_set, n)
    
    # Strategy 3: Local search from best of above
    best = s1 if len(s1) >= len(s2) else s2
    s3 = _local_search(best, n, iterations=5000)
    
    # Pick the best valid set
    candidates = [s1, s2, s3]
    best_set = max(candidates, key=len)
    
    # Final validation
    if not _is_valid_set(best_set):
        # Fallback to primes
        return list(primes)
    
    # Return as sorted list of Python ints
    return sorted(best_set)
# EVOLVE-BLOCK-END
