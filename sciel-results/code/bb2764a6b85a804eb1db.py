"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict

N = 5000
DEADLINE = 325  # seconds, safety margin before 330s kill


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
    """Check if points form an admissible tuple: for every prime p <= len(points),
    some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    # We only need to check primes p <= n
    for p in primes:
        if p > n:
            break
        # Count residue classes
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True


def _greedy_construct(max_points, primes, deadline):
    """
    Greedy construction: start empty, add points one by one.
    For each candidate point, check if adding it keeps the tuple admissible
    for all primes p <= (current_size + 1).
    
    Strategy: try to build a large admissible set by carefully selecting points.
    """
    start_time = time.time()
    
    # We'll build the set incrementally
    # Key insight: for admissibility, for each prime p <= len(S), 
    # we must miss at least one residue class mod p.
    
    # A powerful approach: use the "Chinese remainder" style construction.
    # For a target size m, we need for each prime p <= m, to miss at least one class mod p.
    
    # Let's use a different strategy: start with all numbers 0..N, and remove
    # points to satisfy admissibility constraints.
    
    # Actually, let's think about this differently.
    # The maximum size is bounded. For an admissible tuple of size m in [0, N],
    # for each prime p <= m, we miss at least one residue class mod p.
    # This means at least floor(N/p) + 1 or so points must be removed for each such prime.
    
    # Let's use a greedy removal approach:
    # Start with all points 0, 1, ..., N
    # For each prime p <= N (in increasing order), if all p residue classes are present,
    # remove points from the least-populated residue class.
    
    points = set(range(N + 1))
    
    # Process primes in order of increasing p
    for p in primes:
        if p > N:
            break
        if time.time() - start_time > deadline:
            break
        
        # Count points in each residue class mod p
        class_counts = defaultdict(int)
        for v in points:
            class_counts[v % p] += 1
        
        # If all p classes are present, we need to remove points from one class
        if len(class_counts) == p:
            # Remove from the least populated class
            min_class = min(class_counts, key=lambda c: class_counts[c])
            # Remove all points in that class
            to_remove = [v for v in points if v % p == min_class]
            for v in to_remove:
                points.discard(v)
    
    return list(points)


def _construct_admissible(primes, deadline):
    """
    Construct a large admissible tuple.
    
    Strategy:
    1. Start with all points [0, N]
    2. For each prime p (in increasing order), ensure we miss at least one residue class mod p
    3. When we need to remove a class, remove the one with fewest points to minimize loss
    
    This is the standard greedy approach for admissible tuples.
    """
    start_time = time.time()
    
    # Use a boolean array for efficiency
    present = bytearray(b'\x01') * (N + 1)
    
    # Process primes in increasing order
    for p in primes:
        if p > N:
            break
        if time.time() - start_time > deadline:
            break
        
        # Count points in each residue class mod p
        class_counts = [0] * p
        for v in range(N + 1):
            if present[v]:
                class_counts[v % p] += 1
        
        # If all p classes have at least one point, remove the least populated class
        if all(c > 0 for c in class_counts):
            # Find the class with minimum count
            min_count = min(class_counts)
            # Find all classes with this minimum count (to handle ties)
            min_classes = [c for c in range(p) if class_counts[c] == min_count]
            # Pick the first one (or we could be smarter, but let's keep it simple)
            target_class = min_classes[0]
            
            # Remove all points in this class
            for v in range(target_class, N + 1, p):
                present[v] = 0
    
    # Collect remaining points
    result = [v for v in range(N + 1) if present[v]]
    return result


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    
    An admissible tuple is one where for every prime p <= len(tuple),
    some residue class mod p is missed.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate primes up to N
    primes = _primes(N)
    
    # Use the greedy removal approach
    result = _construct_admissible(primes, deadline - start_time)
    
    # Verify admissibility
    if not _is_admissible(result, primes):
        # Fallback: try a more aggressive removal
        # This shouldn't happen, but just in case
        for p in primes:
            if p > len(result):
                break
            residues = set(v % p for v in result)
            if len(residues) == p:
                # Need to remove more
                class_counts = defaultdict(list)
                for v in result:
                    class_counts[v % p].append(v)
                min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                result = [v for v in result if v % p != min_class]
    
    # Ensure we return distinct integers in [0, N]
    result = sorted(set(result))
    
    return result


# EVOLVE-BLOCK-START
# The main logic is in construct() above
# EVOLVE-BLOCK-END
