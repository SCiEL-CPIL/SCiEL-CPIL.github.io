"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 5000
DEADLINE = 330.0  # wall-clock seconds allowed by evaluator; keep safety margin
SAFETY_MARGIN = 10.0  # return best result before this margin


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


def _is_admissible(points, primes_to_check):
    """Check if points is admissible: for every prime p <= len(points), some residue class mod p is missed."""
    if not points:
        return True
    n = len(points)
    for p in primes_to_check:
        if p > n:
            break
        residues = set(x % p for x in points)
        if len(residues) == p:
            return False
    return True


def _greedy_construct(primes_needed, seed=0):
    """
    Greedy construction: start with all points, remove elements from the most
    populated residue class for each prime that is fully covered.
    """
    rng = random.Random(seed)
    points = list(range(N + 1))
    
    # Precompute primes up to N+1 (max possible tuple size)
    max_p = N + 1
    all_primes = _primes(max_p)
    
    # We need to ensure for each prime p <= len(points), some class is missed
    # Start by ensuring admissibility for all primes up to N+1
    
    changed = True
    while changed:
        changed = False
        current_len = len(points)
        # Only need to check primes <= current_len
        for p in all_primes:
            if p > current_len:
                break
            # Count residues
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            if all(c > 0 for c in counts):
                # Find the smallest class to remove
                min_idx = min(range(p), key=lambda i: counts[i])
                # Remove all points in that class
                points = [v for v in points if v % p != min_idx]
                changed = True
                break
    
    return points


def _local_search_add(points, primes_to_check, seed=0, max_iterations=10000):
    """
    Try to add back removed elements one at a time if it doesn't violate admissibility.
    """
    rng = random.Random(seed)
    points_set = set(points)
    candidates = [x for x in range(N + 1) if x not in points_set]
    rng.shuffle(candidates)
    
    improved = False
    for x in candidates:
        new_points = list(points) + [x]
        if _is_admissible(new_points, primes_to_check):
            points = new_points
            points_set.add(x)
            improved = True
    
    return points, improved


def _remove_minority(points, primes_to_check, seed=0):
    """
    For each prime that is fully covered, remove the minority class.
    Returns new points and whether any removal happened.
    """
    changed = False
    current_len = len(points)
    for p in primes_to_check:
        if p > current_len:
            break
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        if all(c > 0 for c in counts):
            min_idx = min(range(p), key=lambda i: counts[i])
            points = [v for v in points if v % p != min_idx]
            changed = True
    return points, changed


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Get primes up to N+1
    all_primes = _primes(N + 1)
    
    best_points = None
    best_size = 0
    
    # Strategy 1: Greedy construction with different seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        
        points = _greedy_construct(all_primes, seed=seed)
        
        # Local search: try to add back elements
        # Determine which primes we need to check
        p_limit = len(points)
        primes_to_check = [p for p in all_primes if p <= p_limit]
        
        # Multiple rounds of local search
        for round_num in range(5):
            if time.time() > deadline:
                break
            points, improved = _local_search_add(points, primes_to_check, seed=seed * 1000 + round_num)
            if not improved:
                break
            # Update primes_to_check as size may have changed
            p_limit = len(points)
            primes_to_check = [p for p in all_primes if p <= p_limit]
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = list(points)
    
    # Strategy 2: Start from a smaller set and grow
    for seed in range(50):
        if time.time() > deadline:
            break
        
        # Start with a random subset
        rng = random.Random(seed + 1000)
        subset_size = max(10, N // 10)
        points = rng.sample(range(N + 1), subset_size)
        
        # Make it admissible by removing minority classes
        changed = True
        while changed:
            if time.time() > deadline:
                break
            points, changed = _remove_minority(points, all_primes, seed=seed)
        
        # Local search to add back
        p_limit = len(points)
        primes_to_check = [p for p in all_primes if p <= p_limit]
        
        for round_num in range(5):
            if time.time() > deadline:
                break
            points, improved = _local_search_add(points, primes_to_check, seed=seed * 1000 + round_num + 500)
            if not improved:
                break
            p_limit = len(points)
            primes_to_check = [p for p in all_primes if p <= p_limit]
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = list(points)
    
    # Strategy 3: Start from primes above sqrt(N) and grow
    for seed in range(50):
        if time.time() > deadline:
            break
        
        # Start with 1 and primes above sqrt(N)
        sqrt_n = int(N ** 0.5) + 1
        points = [1] + [p for p in all_primes if p * p > N]
        
        # Make admissible
        changed = True
        while changed:
            if time.time() > deadline:
                break
            points, changed = _remove_minority(points, all_primes, seed=seed)
        
        # Local search
        p_limit = len(points)
        primes_to_check = [p for p in all_primes if p <= p_limit]
        
        for round_num in range(10):
            if time.time() > deadline:
                break
            points, improved = _local_search_add(points, primes_to_check, seed=seed * 1000 + round_num + 1000)
            if not improved:
                break
            p_limit = len(points)
            primes_to_check = [p for p in all_primes if p <= p_limit]
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = list(points)
    
    # Final check: ensure best_points is admissible
    if best_points is None:
        # Fallback: return a simple admissible set
        points = list(range(N + 1))
        changed = True
        while changed:
            points, changed = _remove_minority(points, all_primes, seed=0)
        best_points = points
    else:
        # Verify admissibility
        p_limit = len(best_points)
        primes_to_check = [p for p in all_primes if p <= p_limit]
        if not _is_admissible(best_points, primes_to_check):
            # Repair
            changed = True
            while changed:
                best_points, changed = _remove_minority(best_points, all_primes, seed=42)
    
    return best_points


# EVOLVE-BLOCK-START
# (The above functions are the actual implementation; this block is for structural compatibility)
# EVOLVE-BLOCK-END
