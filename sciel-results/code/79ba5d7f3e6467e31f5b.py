"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import product

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple for the given primes."""
    n = len(points)
    if n == 0:
        return True
    for p in primes_to_check:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True

def _get_smallest_violated_prime(points, primes):
    """Find the smallest prime p <= len(points) where all residues mod p are covered."""
    n = len(points)
    for p in primes:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return p
    return None

def _greedy_remove(points, primes):
    """
    Greedily repair admissibility by removing elements from the least-populated
    residue class for the smallest violated prime.
    """
    points = list(points)
    changed = True
    while changed:
        changed = False
        n = len(points)
        p = _get_smallest_violated_prime(points, primes)
        if p is None:
            break
        # Count residues mod p
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        min_count = min(counts)
        # Remove one element from each residue class with minimum count
        # Actually, remove all elements from the least populated residue class
        target_residues = [r for r in range(p) if counts[r] == min_count]
        # Remove elements from the first target residue
        target = target_residues[0]
        points = [v for v in points if v % p != target]
        changed = True
    return points

def _greedy_add(points, primes, deadline):
    """
    Try to add back elements from [0, N] that were removed, if they maintain admissibility.
    """
    points = list(points)
    point_set = set(points)
    all_candidates = [v for v in range(N + 1) if v not in point_set]
    
    # Sort candidates by some heuristic - try adding elements that are "safe"
    # For each candidate, check if adding it keeps the tuple admissible
    added = []
    for v in all_candidates:
        if time.time() > deadline:
            break
        test_points = points + [v]
        n = len(test_points)
        # Quick check: for each prime p <= n, check if v creates a violation
        violated = False
        for p in primes:
            if p > n:
                break
            # Check if adding v completes all residues mod p
            residues = set(test_points[i] % p for i in range(len(test_points)))
            if len(residues) == p:
                violated = True
                break
        if not violated:
            points.append(v)
            point_set.add(v)
            added.append(v)
    
    return points, added

def _local_search(points, primes, deadline):
    """
    Local search: try swapping elements to improve the solution.
    We try to replace one element with another to maintain admissibility while potentially
    allowing more elements to be added later.
    """
    points = list(points)
    point_set = set(points)
    all_candidates = [v for v in range(N + 1) if v not in point_set]
    
    best_points = points
    best_size = len(points)
    
    # Try random swaps
    import random
    rng = random.Random(42)
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        if not all_candidates or not points:
            break
        
        # Pick a random element to remove
        idx = rng.randrange(len(points))
        removed = points[idx]
        
        # Pick a random candidate to add
        cand_idx = rng.randrange(len(all_candidates))
        candidate = all_candidates[cand_idx]
        
        # Try the swap
        test_points = points[:idx] + points[idx+1:] + [candidate]
        n = len(test_points)
        
        # Check admissibility
        admissible = True
        for p in primes:
            if p > n:
                break
            residues = set(v % p for v in test_points)
            if len(residues) == p:
                admissible = False
                break
        
        if admissible and len(test_points) > best_size:
            best_points = test_points
            best_size = len(test_points)
            points = test_points
            point_set = set(points)
            all_candidates = [v for v in range(N + 1) if v not in point_set]
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10 second safety margin
    
    # Get all primes up to N
    all_primes = _primes(N)
    
    # Start with the full set [0, N]
    points = list(range(N + 1))
    
    # Phase 1: Greedy removal to make it admissible
    points = _greedy_remove(points, all_primes)
    best_points = points
    best_size = len(points)
    
    # Phase 2: Greedy add - try to add back elements
    if time.time() < deadline - 5:
        points_added, _ = _greedy_add(points, all_primes, deadline - 5)
        if len(points_added) > best_size:
            best_points = points_added
            best_size = len(points_added)
    
    # Phase 3: Local search
    if time.time() < deadline - 2:
        points_ls = _local_search(best_points, all_primes, deadline - 2)
        if len(points_ls) > best_size:
            best_points = points_ls
            best_size = len(points_ls)
    
    # Verify admissibility of the best result
    n = len(best_points)
    relevant_primes = [p for p in all_primes if p <= n]
    if not _is_admissible(best_points, relevant_primes):
        # If somehow not admissible, do a final repair
        best_points = _greedy_remove(best_points, all_primes)
    
    return best_points

# EVOLVE-BLOCK-END
