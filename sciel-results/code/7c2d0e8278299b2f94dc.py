"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

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

def _can_add(candidate, current_set):
    """Check if candidate can be added to current_set without violating the constraint.
    For every three distinct members a,b,c in the set, a must not divide b*c."""
    cs = list(current_set)
    n = len(cs)
    
    # Case 1: candidate plays role of 'a': candidate ∤ b*c for all distinct b,c in current_set
    for i in range(n):
        for j in range(i+1, n):
            if (cs[i] * cs[j]) % candidate == 0:
                return False
    
    # Case 2: candidate plays role of 'b' (or 'c'): for each a in current_set, a ∤ candidate * b
    # where b is another distinct element from current_set
    for i in range(n):
        a = cs[i]
        for j in range(n):
            if j == i:
                continue
            b = cs[j]
            if (candidate * b) % a == 0:
                return False
    return True

def _greedy_construct():
    """Greedy construction starting from primes."""
    primes = [a for a in range(2, P['n']+1) if _is_prime(a)]
    current = set(primes)
    
    # Try adding composites in increasing order
    composites = [a for a in range(2, P['n']+1) if not _is_prime(a)]
    for cand in composites:
        if _can_add(cand, current):
            current.add(cand)
    return current

def _local_search(initial_set, deadline):
    """Local search with 1-swap moves."""
    best = set(initial_set)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for elem in list(best):
            if time.time() > deadline - 3:
                break
            trial = best - {elem}
            for cand in range(2, P['n']+1):
                if cand in trial:
                    continue
                if _can_add(cand, trial):
                    new_set = trial | {cand}
                    if len(new_set) > len(best):
                        best = new_set
                        improved = True
                        break
    return best

def construct():
    deadline = time.time() + 50  # safety margin
    
    # Phase 1: Greedy construction
    greedy_set = _greedy_construct()
    
    # Phase 2: Local search improvement
    result = _local_search(greedy_set, deadline)
    
    return [int(x) for x in sorted(result)]
# EVOLVE-BLOCK-END
