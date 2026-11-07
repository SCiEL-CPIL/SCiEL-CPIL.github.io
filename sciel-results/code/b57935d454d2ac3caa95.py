"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 256}

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

def _check_valid(S):
    """Check if set S is valid: for all distinct a,b,c in S, a does not divide b*c."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S_list[i], S_list[j], S_list[k]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 50  # 50s safety margin
    
    n_max = P['n']
    
    # Get all primes
    primes = [x for x in range(2, n_max + 1) if _is_prime(x)]
    
    # Start with all primes
    current = set(primes)
    
    def can_add(candidate, S):
        """Check if adding candidate to S maintains validity."""
        S_list = list(S)
        
        # Check: candidate does not divide b*c for any pair b,c in S
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                b, c = S_list[i], S_list[j]
                if (b * c) % candidate == 0:
                    return False
        
        # Check: for each pair b,c in S, b does not divide candidate*c and c does not divide candidate*b
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                b, c = S_list[i], S_list[j]
                if (candidate * c) % b == 0:
                    return False
                if (candidate * b) % c == 0:
                    return False
        return True
    
    # Get composites
    composites = [x for x in range(2, n_max + 1) if not _is_prime(x)]
    
    # Try multiple passes with different orderings
    orderings = [
        sorted(composites),
        sorted(composites, reverse=True),
        sorted(composites, key=lambda x: x.bit_count()),
        sorted(composites, key=lambda x: -x.bit_count()),
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        for comp in ordering:
            if comp not in current and can_add(comp, current):
                current.add(comp)
    
    # Local search: try removing one element and adding two
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for remove_elem in list(current):
            if time.time() > deadline:
                break
            temp_set = current - {remove_elem}
            added = 0
            for add_elem in composites:
                if add_elem in temp_set:
                    continue
                if can_add(add_elem, temp_set):
                    temp_set = temp_set | {add_elem}
                    added += 1
                    if added >= 2:
                        break
            if len(temp_set) > len(current):
                current = temp_set
                improved = True
    
    # Another greedy pass
    for comp in sorted(composites):
        if time.time() > deadline:
            break
        if comp not in current and can_add(comp, current):
            current.add(comp)
    
    for comp in sorted(composites, reverse=True):
        if time.time() > deadline:
            break
        if comp not in current and can_add(comp, current):
            current.add(comp)
    
    result = sorted(current)
    
    # Verify validity
    if not _check_valid(set(result)):
        # Fallback to primes only
        result = primes
    
    # Ensure we return Python ints in a list
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
