"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000

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


def _compute_empty_residues(points, primes_needed):
    """
    For each prime p in primes_needed, compute which residue classes mod p are empty.
    Returns a dict: p -> set of empty residue classes.
    """
    result = {}
    for p in primes_needed:
        seen = [False] * p
        for v in points:
            seen[v % p] = True
        empty = set()
        for r in range(p):
            if not seen[r]:
                empty.add(r)
        result[p] = empty
    return result


def _check_admissible(points, primes_needed):
    """
    Check if points is admissible: for every prime p <= len(points), some residue class mod p is missed.
    Returns True if admissible, False otherwise.
    """
    n = len(points)
    # Only need to check primes <= n
    for p in primes_needed:
        if p > n:
            break
        seen = [False] * p
        for v in points:
            seen[v % p] = True
        if all(seen):
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 320  # 320s deadline, 10s margin
    
    # Strategy: Start with all integers [0, N], then greedily remove elements
    # to satisfy admissibility constraints, prioritizing removals that minimize
    # the impact on future constraints.
    
    # We need to ensure that for each prime p <= len(tuple), at least one residue
    # class mod p is empty.
    
    # Start with all points
    points = set(range(N + 1))
    
    # We'll iteratively remove elements from the most "constrained" primes
    # A prime p is constrained if all residue classes mod p are currently occupied
    
    # Get all primes up to N
    all_primes = _primes(N)
    
    # We'll use a priority queue approach: process primes in order of how "close"
    # they are to being fully covered
    # For each prime p, if all p residue classes are occupied, we must remove at least one class
    
    # Let's iterate: find the smallest prime p where all residue classes are occupied,
    # and remove the least-populated residue class for that prime
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        n = len(points)
        if n == 0:
            break
        
        # Find primes <= n that have all residue classes occupied
        violating_primes = []
        for p in all_primes:
            if p > n:
                break
            seen = [False] * p
            for v in points:
                seen[v % p] = True
            if all(seen):
                violating_primes.append(p)
        
        if not violating_primes:
            break
        
        # Choose the most restrictive prime (smallest p, or the one where removal hurts least)
        # Let's choose the smallest violating prime first
        p = min(violating_primes)
        
        # Find the residue class mod p with the fewest elements
        class_counts = {}
        for v in points:
            r = v % p
            if r not in class_counts:
                class_counts[r] = 0
            class_counts[r] += 1
        
        # Remove the least-populated class
        min_class = min(class_counts, key=lambda r: class_counts[r])
        to_remove = {v for v in points if v % p == min_class}
        points -= to_remove
        changed = True
    
    # Now points should be admissible. Let's verify.
    # But we can also try to add back some elements if possible
    
    # Try to add back elements that were removed, in an order that maintains admissibility
    # This is complex, so let's just return what we have
    
    result = sorted(points)
    
    # Verify admissibility
    if not _check_admissible(result, all_primes):
        # Fallback: use the seed approach
        points_seed = [1] + [p for p in all_primes if p * p > N]
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for p in all_primes:
                if p > len(points_seed):
                    break
                classes = {}
                for v in points_seed:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points_seed = [v for v in points_seed if v not in smallest]
                    changed = True
                    break
        result = points_seed
    
    return result
# EVOLVE-BLOCK-END
