"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

N = 5000
DEADLINE = 320  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_needed):
    """Check if points is admissible w.r.t. all primes in primes_needed."""
    for p in primes_needed:
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True


def _greedy_construct():
    """
    Start from 1 and the primes above sqrt(N); repair any prime whose residue
    classes are all covered by deleting the least-populated class, until admissible.
    """
    points = [1] + [p for p in _primes(N) if p * p > N]
    changed = True
    while changed:
        changed = False
        for p in _primes(len(points)):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                changed = True
                break
    return points


def _local_search_expansion(points, primes_needed, deadline):
    """
    Try to add back elements that were removed. For each candidate not in points,
    check if adding it keeps the tuple admissible. Use a greedy approach with
    multiple passes to find a larger admissible set.
    """
    points = set(points)
    all_candidates = set(range(N + 1)) - points
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle candidates for diversity
        candidates = list(all_candidates)
        random.shuffle(candidates)
        
        for v in candidates:
            if time.time() >= deadline:
                break
            if v in points:
                continue
            
            new_points = points | {v}
            if _is_admissible(list(new_points), primes_needed):
                points.add(v)
                all_candidates.discard(v)
                improved = True
    
    return list(points)


def _local_search_deletion(points, primes_needed, deadline):
    """
    Try to remove elements that are "redundant" in the sense that removing them
    still leaves the tuple admissible, then try to add back other elements.
    This is a more aggressive local search.
    """
    points = set(points)
    
    # First, try to find elements that can be removed without breaking admissibility
    # An element can be removed if for every prime p, the element's residue class
    # is still covered by at least one other element in the set.
    
    removable = []
    for v in list(points):
        if time.time() >= deadline:
            break
        can_remove = True
        for p in primes_needed:
            r = v % p
            # Check if any other element in points has residue r mod p
            found = False
            for u in points:
                if u != v and u % p == r:
                    found = True
                    break
            if not found:
                can_remove = False
                break
        if can_remove:
            removable.append(v)
    
    # Try removing one element at a time, then expanding
    best_set = set(points)
    for v in removable:
        if time.time() >= deadline:
            break
        new_points = set(points)
        new_points.discard(v)
        # Now try to expand from this smaller set
        expanded = _local_search_expansion(list(new_points), primes_needed, deadline)
        if len(expanded) > len(best_set):
            best_set = set(expanded)
    
    return list(best_set)


def construct():
    start_time = time.time()
    
    # Step 1: Greedy construction
    points = _greedy_construct()
    primes_needed = _primes(len(points))
    
    # Step 2: Local search expansion - try to add back elements
    points = _local_search_expansion(points, primes_needed, DEADLINE)
    primes_needed = _primes(len(points))
    
    # Step 3: More aggressive local search with deletion
    if time.time() < DEADLINE:
        points = _local_search_deletion(points, primes_needed, DEADLINE)
        primes_needed = _primes(len(points))
    
    # Final expansion pass
    if time.time() < DEADLINE:
        points = _local_search_expansion(points, primes_needed, DEADLINE)
        primes_needed = _primes(len(points))
    
    # Ensure admissibility
    if not _is_admissible(points, primes_needed):
        # Fallback: use greedy result
        points = _greedy_construct()
    
    return points
# EVOLVE-BLOCK-END
