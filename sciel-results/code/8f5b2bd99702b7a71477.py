"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 5000
MAX_TIME = 290.0  # Safety margin before the 300s limit

# EVOLVE-BLOCK-START

def _primes_up_to(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p:limit + 1:p] = bytearray(len(sieve[p * p:limit + 1:p]))
    return [p for p in range(limit + 1) if sieve[p]]

PRIMES = _primes_up_to(N + 1)
PRIME_SET = set(PRIMES)


def _is_admissible(s):
    """Check if set s is admissible: for every prime p <= len(s), some residue mod p is missed."""
    k = len(s)
    for p in PRIMES:
        if p > k:
            break
        residues = set(x % p for x in s)
        if len(residues) == p:
            return False
    return True


def _greedy_add(s, candidates, deadline):
    """Try to add candidates to s (a set) in order, maintaining admissibility."""
    s = set(s)
    k = len(s)
    # Precompute which primes are relevant
    relevant_primes = [p for p in PRIMES if p <= k + 5]  # a few ahead
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in s:
            continue
        new_k = k + 1
        new_set = s | {x}
        
        # Check admissibility for all primes p <= new_k
        valid = True
        for p in PRIMES:
            if p > new_k:
                break
            # Count residues
            residues = set()
            for val in new_set:
                residues.add(val % p)
                if len(residues) == p:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s = new_set
            k = new_k
    
    return s


def _random_grow(deadline):
    """Randomized greedy growth: start with a small random set, add elements."""
    # Start with a random seed of size 5-15
    start_size = random.randint(5, 15)
    start_set = set(random.sample(range(N + 1), start_size))
    
    # Verify start is admissible
    if not _is_admissible(start_set):
        # Try a few more times
        for _ in range(10):
            start_set = set(random.sample(range(N + 1), start_size))
            if _is_admissible(start_set):
                break
    
    # Generate candidates in random order
    candidates = list(range(N + 1))
    random.shuffle(candidates)
    
    result = _greedy_add(start_set, candidates, deadline)
    return result


def _systematic_grow(deadline):
    """Systematic growth: try to add elements in increasing order."""
    start_set = set()
    candidates = list(range(N + 1))
    return _greedy_add(start_set, candidates, deadline)


def _local_search(s, deadline):
    """Try to improve set s by swapping elements."""
    s = set(s)
    k = len(s)
    improvement_found = True
    while improvement_found and time.time() < deadline:
        improvement_found = False
        # Try removing each element and adding a different one
        for x in list(s):
            if time.time() > deadline:
                break
            s_minus = s - {x}
            new_k = k  # same size
            # Try to add a new element
            for y in range(N + 1):
                if y in s_minus:
                    continue
                candidate_set = s_minus | {y}
                if len(candidate_set) != k:
                    continue
                # Check admissibility
                valid = True
                for p in PRIMES:
                    if p > k:
                        break
                    residues = set(val % p for val in candidate_set)
                    if len(residues) == p:
                        valid = False
                        break
                if valid:
                    s = candidate_set
                    improvement_found = True
                    break
    return s


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    best = set()
    
    # Phase 1: Multiple random restarts with greedy growth
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        result = _random_grow(deadline - 5)
        if len(result) > len(best):
            best = result
    
    # Phase 2: Try systematic growth
    if time.time() < deadline - 10:
        result = _systematic_grow(deadline - 5)
        if len(result) > len(best):
            best = result
    
    # Phase 3: Local search on best found
    if time.time() < deadline - 5 and len(best) > 0:
        best = _local_search(best, deadline - 2)
    
    # Phase 4: Additional random restarts if time remains
    while time.time() < deadline - 2:
        result = _random_grow(deadline - 1)
        if len(result) > len(best):
            best = result
    
    return list(best)

# EVOLVE-BLOCK-END
