"""Improved admissible tuple construction for the fixed-diameter maximization task."""
import time
import math
import random
from collections import defaultdict

N = 5000
DEADLINE_SECONDS = 320.0  # safety margin before 330s kill

def _primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [i for i in range(limit + 1) if sieve[i]]

def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple: for every prime p in primes_to_check,
    some residue class mod p is missed."""
    for p in primes_to_check:
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True

def _greedy_removal(points, primes_to_check, deadline):
    """Iteratively remove the least-populated residue class for any fully-covered prime."""
    points = list(points)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for p in primes_to_check:
            classes = defaultdict(list)
            for v in points:
                classes[v % p].append(v)
            if len(classes) == p:
                # Find smallest class to remove
                smallest = min(classes.values(), key=len)
                smallest_set = set(smallest)
                points = [v for v in points if v not in smallest_set]
                changed = True
                break
    return points

def _add_back_search(points, primes_to_check, deadline, max_iters=50000):
    """Try to add back removed elements to increase tuple size while maintaining admissibility."""
    points = list(points)
    point_set = set(points)
    # Candidates: elements in [0, N] not in points
    candidates = [v for v in range(N + 1) if v not in point_set]
    random.shuffle(candidates)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if time.time() > deadline:
            break
        improved = False
        iters += 1
        for v in candidates:
            if time.time() > deadline:
                break
            new_points = points + [v]
            new_set = point_set | {v}
            # Check admissibility
            admissible = True
            for p in primes_to_check:
                residues = set()
                for x in new_points:
                    residues.add(x % p)
                if len(residues) == p:
                    admissible = False
                    break
            if admissible:
                points = new_points
                point_set = new_set
                candidates.remove(v)
                improved = True
    return points

def _local_optimization(points, primes_to_check, deadline):
    """More systematic local search: try swapping elements."""
    points = list(points)
    point_set = set(points)
    candidates = [v for v in range(N + 1) if v not in point_set]
    
    # Try adding one element at a time
    added = True
    while added and time.time() < deadline:
        added = False
        for v in candidates[:]:
            if time.time() > deadline:
                break
            new_set = point_set | {v}
            admissible = True
            for p in primes_to_check:
                residues = set(x % p for x in new_set)
                if len(residues) == p:
                    admissible = False
                    break
            if admissible:
                points.append(v)
                point_set = new_set
                candidates.remove(v)
                added = True
    return points

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Primes up to len(tuple) - we need primes up to N since tuple could be up to N+1 elements
    # Actually, for admissibility, we need to check primes p <= len(tuple)
    # Since len(tuple) <= N+1 = 5001, we need primes up to 5001
    all_primes = _primes(N + 1)
    
    best_points = []
    best_size = 0
    
    # Strategy 1: Start with all integers 0..N, remove to make admissible
    # This is the densest starting point
    points = list(range(N + 1))
    points = _greedy_removal(points, all_primes, deadline)
    points = _local_optimization(points, all_primes, deadline)
    best_points = points
    best_size = len(points)
    
    if time.time() > deadline:
        return best_points
    
    # Strategy 2: Multi-strategy with different initializations
    random.seed(42)
    
    # Try different random subsets as starting points
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Random subset of size ~3000
        subset_size = random.randint(2500, 4000)
        init_points = random.sample(range(N + 1), subset_size)
        init_points = _greedy_removal(init_points, all_primes, deadline)
        init_points = _local_optimization(init_points, all_primes, deadline)
        
        if len(init_points) > best_size:
            best_points = init_points
            best_size = len(init_points)
        
        if time.time() > deadline - 10:
            break
    
    # Final: try to add back from best_points
    if time.time() < deadline - 5:
        best_points = _add_back_search(best_points, all_primes, deadline)
        best_size = len(best_points)
    
    # Ensure we return a valid admissible tuple
    # Verify admissibility
    primes_to_check = [p for p in all_primes if p <= len(best_points)]
    if not _is_admissible(best_points, primes_to_check):
        # Fallback: use greedy removal from full set
        points = list(range(N + 1))
        points = _greedy_removal(points, all_primes, deadline)
        best_points = points
    
    return best_points
