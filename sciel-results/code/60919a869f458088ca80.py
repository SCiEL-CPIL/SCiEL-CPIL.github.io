"""Construct the largest admissible tuple within [0, 5000]."""
import time
from collections import defaultdict

N = 5000
DEADLINE = 325  # safety margin before 330s kill


def _primes(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes):
    """Check admissibility: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    for p in primes:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _construct_greedy(primes, deadline):
    """
    Greedy removal: start with all points [0, N]. For each prime p in increasing order,
    if all p residue classes are still present, remove the least-populated class.
    This creates overlap: early deletions destroy classes for later primes.
    """
    start = time.time()
    present = bytearray(b'\x01') * (N + 1)
    
    for p in primes:
        if p > N:
            break
        if time.time() - start > deadline:
            break
        
        # Count points in each residue class mod p
        class_counts = [0] * p
        for v in range(N + 1):
            if present[v]:
                class_counts[v % p] += 1
        
        # If all p classes have at least one point, remove the least populated class
        if min(class_counts) > 0:
            min_count = min(class_counts)
            # Find the first class with minimum count
            target_class = class_counts.index(min_count)
            
            # Remove all points in this class
            for v in range(target_class, N + 1, p):
                present[v] = 0
    
    # Collect remaining points
    result = [v for v in range(N + 1) if present[v]]
    return result


def _refine(points, primes, deadline):
    """
    Try to recover removed elements that don't violate admissibility.
    For each removed element, check if adding it back keeps the tuple admissible.
    Process in order of smallest value first (arbitrary heuristic).
    """
    start = time.time()
    if not points:
        return points
    
    points_set = set(points)
    current_len = len(points)
    
    # Generate list of removed elements
    removed = [v for v in range(N + 1) if v not in points_set]
    
    # Try to add back removed elements one by one
    for v in removed:
        if time.time() - start > deadline:
            break
        
        # Check if adding v keeps the tuple admissible
        new_points = points_set | {v}
        new_len = len(new_points)
        
        # Check admissibility for all primes p <= new_len
        admissible = True
        for p in primes:
            if p > new_len:
                break
            residues = set(x % p for x in new_points)
            if len(residues) == p:
                admissible = False
                break
        
        if admissible:
            points_set.add(v)
    
    return list(points_set)


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    
    An admissible tuple is one where for every prime p <= len(tuple),
    some residue class mod p is missed.
    """
    start_time = time.time()
    
    # Generate primes up to N
    primes = _primes(N)
    
    # Phase 1: Greedy removal in increasing prime order
    deadline_phase1 = DEADLINE * 0.7  # Use 70% of time for initial construction
    result = _construct_greedy(primes, deadline_phase1)
    
    # Verify admissibility
    if not _is_admissible(result, primes):
        # Fallback: aggressive removal
        for p in primes:
            if p > len(result):
                break
            residues = set(v % p for v in result)
            if len(residues) == p:
                class_counts = defaultdict(list)
                for v in result:
                    class_counts[v % p].append(v)
                min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                result = [v for v in result if v % p != min_class]
    
    # Phase 2: Refinement - try to recover removed elements
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 5  # Leave 5s safety margin
    if remaining_time > 10:
        result = _refine(result, primes, remaining_time)
    
    # Final verification
    if not _is_admissible(result, primes):
        # Emergency fallback: remove until admissible
        for p in primes:
            if p > len(result):
                break
            residues = set(v % p for v in result)
            if len(residues) == p:
                class_counts = defaultdict(list)
                for v in result:
                    class_counts[v % p].append(v)
                min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                result = [v for v in result if v % p != min_class]
    
    # Ensure we return distinct integers in [0, N]
    result = sorted(set(result))
    
    return result
