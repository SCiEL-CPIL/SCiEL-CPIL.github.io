"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

N = 5000
DEADLINE = 320  # wall-clock seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes):
    """Check if points is admissible: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    for p in primes:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True

def _greedy_repair(points, primes, max_iters=5000):
    """Greedy repair: iteratively remove the least-populated residue class for the first offending prime."""
    points = list(points)
    for _ in range(max_iters):
        n = len(points)
        offending = None
        for p in primes:
            if p > n:
                break
            residues = [v % p for v in points]
            if len(set(residues)) == p:
                offending = p
                break
        if offending is None:
            break
        p = offending
        classes = {}
        for v in points:
            r = v % p
            if r not in classes:
                classes[r] = []
            classes[r].append(v)
        # Remove the smallest class
        min_class = min(classes.values(), key=len)
        min_set = set(min_class)
        points = [v for v in points if v not in min_set]
    return points

def _score(points):
    return len(points)

def construct():
    start_time = time.time()
    
    # Primes up to N
    all_primes = _primes(N)
    
    # Strategy: Start with a good initial set, then try to improve via local search
    # Initial set: all integers in [0, N] that are NOT divisible by any small prime
    # This gives a large admissible set
    
    # Actually, let's think about this differently.
    # We want the LARGEST admissible tuple in [0, N].
    # Admissible means: for every prime p <= k (where k = len(tuple)), there's a residue class mod p not in the tuple.
    
    # Key insight: If we take all numbers in [0, N] except those in certain residue classes,
    # we need to ensure that for each prime p <= k, at least one residue class mod p is completely absent.
    
    # A simple approach: For each prime p <= N, decide which residue class to remove.
    # But we want to maximize the size, so we want to remove as few numbers as possible.
    
    # Let's try: Start with all numbers [0, N], then iteratively remove residue classes
    # to make it admissible, choosing removals that minimize the impact on size.
    
    # Alternative: Start with a known admissible set and try to add elements.
    
    # Let's try a different approach: 
    # For each prime p <= N, we must miss at least one residue class mod p.
    # The constraint is that for primes p > len(tuple), we don't need to miss anything.
    # So if we make the tuple large enough that len(tuple) < smallest_prime_we_care_about,
    # wait no, len(tuple) can be up to N+1 = 5001.
    
    # Let's think: if k = len(tuple), we need for all primes p <= k, to miss a residue class mod p.
    # The primes up to 5001 are about 669 primes.
    
    # Strategy: Choose for each prime p a residue class r_p to exclude.
    # Then the tuple is {x in [0,N] : for all primes p <= k, x % p != r_p}.
    # But k depends on the size of this set, which is circular.
    
    # Let's try: assume k is around some value, say K. Then for all primes p <= K, pick a residue to exclude.
    # The size of the resulting set is approximately (N+1) * prod_{p<=K} (1 - 1/p).
    # We want to maximize this, but also need the actual size to be consistent with K.
    
    # Let's try a concrete approach:
    # 1. Start with all numbers in [0, N].
    # 2. For each prime p in increasing order, if all p residue classes are present, remove the smallest class.
    # 3. This gives an admissible set.
    # 4. Try to improve by local search.
    
    best_points = None
    best_size = 0
    
    # Phase 1: Greedy construction
    points = list(range(N + 1))
    points = _greedy_repair(points, all_primes, max_iters=10000)
    if _is_admissible(points, all_primes):
        best_points = points
        best_size = len(points)
    
    if time.time() - start_time > DEADLINE - 10:
        return best_points if best_points else []
    
    # Phase 2: Try to add back some elements that were removed
    # Local search: try to add single elements that maintain admissibility
    
    if best_points is None:
        return []
    
    current = set(best_points)
    current_size = len(current)
    
    # Try random additions
    removed = set(range(N + 1)) - current
    removed_list = list(removed)
    random.shuffle(removed_list)
    
    attempts = 0
    max_attempts = 50000
    for x in removed_list:
        if time.time() - start_time > DEADLINE - 5:
            break
        if attempts >= max_attempts:
            break
        attempts += 1
        
        # Try adding x
        new_size = current_size + 1
        # Check if adding x maintains admissibility for all primes p <= new_size
        # We only need to check primes p where the residue class of x might complete all p classes
        added = True
        for p in all_primes:
            if p > new_size:
                break
            # Check if x completes all p residue classes
            residues = set((v % p) for v in current)
            if len(residues) < p:  # Already missing some class, so adding x can't make it complete unless it fills the last gap
                # Actually we need to check: after adding x, do we have all p classes?
                if x % p not in residues:
                    # x adds a new class, but we still need to check if now all p are present
                    if len(residues) + 1 == p:
                        # Adding x would complete all p classes -> not admissible
                        added = False
                        break
                # else x is in an existing class, no change in coverage
            else:
                # All p classes already present before adding x -> already not admissible (shouldn't happen)
                added = False
                break
        if added:
            current.add(x)
            current_size += 1
            best_size = current_size
    
    if current_size > best_size:
        best_points = list(current)
        best_size = current_size
    
    if time.time() - start_time > DEADLINE - 3:
        return best_points
    
    # Phase 3: Try removing and re-adding different sets (swap)
    # For each element in current, try swapping it with a removed element
    current_set = set(best_points)
    current_size = len(current_set)
    removed_set = set(range(N + 1)) - current_set
    removed_list = list(removed_set)
    random.shuffle(removed_list)
    
    current_list = list(current_set)
    random.shuffle(current_list)
    
    swaps = 0
    max_swaps = 30000
    for i in range(min(len(current_list), 2000)):
        if time.time() - start_time > DEADLINE - 3:
            break
        if swaps >= max_swaps:
            break
        x = current_list[i]
        for j in range(min(len(removed_list), 100)):
            if time.time() - start_time > DEADLINE - 3:
                break
            y = removed_list[j]
            swaps += 1
            
            # Try swapping x for y
            trial = current_set - {x} | {y}
            trial_size = len(trial)
            
            # Check admissibility
            admissible = True
            for p in all_primes:
                if p > trial_size:
                    break
                residues = set(v % p for v in trial)
                if len(residues) == p:
                    admissible = False
                    break
            if admissible:
                current_set = trial
                current_size = trial_size
                best_size = current_size
                best_points = list(current_set)
                # Update removed_list
                removed_set = set(range(N + 1)) - current_set
                removed_list = list(removed_set)
                random.shuffle(removed_list)
                break
    
    return best_points

# EVOLVE-BLOCK-END
