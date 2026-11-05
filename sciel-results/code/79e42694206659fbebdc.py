"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_list):
    """Check if points form an admissible tuple: for every prime p <= len(points),
    some residue class mod p is missed."""
    n = len(points)
    for p in primes_list:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _greedy_remove(points, primes, deadline):
    """Greedily remove residue classes, processing primes in decreasing order."""
    points_set = set(points)
    points_list = list(points)
    
    # Process primes in decreasing order (larger primes first)
    # This removes larger chunks while preserving more points
    for p in reversed(primes):
        if time.time() > deadline:
            break
        if p > len(points_list):
            continue
        
        # Count residue classes efficiently
        counts = [0] * p
        for v in points_list:
            counts[v % p] += 1
        
        # Check if all classes are covered
        if min(counts) > 0:
            # All p classes are present; remove the least-populated class
            min_count = min(counts)
            min_class = counts.index(min_count)
            
            # Remove all points in this residue class
            points_list = [v for v in points_list if v % p != min_class]
            points_set = set(points_list)
    
    return points_list


def _local_improve(points, primes, deadline):
    """Try to add back points that were removed, if it maintains admissibility."""
    if not points:
        return points
    
    points_set = set(points)
    points_list = list(points)
    n_points = N + 1  # total possible points
    
    # Try adding back points in random-ish order (use modular pattern for diversity)
    # We'll try candidates from 0 to N that aren't in our set
    candidates = [v for v in range(N + 1) if v not in points_set]
    
    # Sort candidates by some heuristic - try to add points that help with sparse residue classes
    # For now, just iterate through them and try adding
    added = 0
    for v in candidates:
        if time.time() > deadline:
            break
        
        # Try adding v
        new_points = points_list + [v]
        if _is_admissible(new_points, primes):
            points_list = new_points
            points_set.add(v)
            added += 1
    
    return points_list


def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 300s limit
    
    # Get primes up to N+1 (since len(points) can be up to N+1)
    primes = _primes(N + 1)
    
    # Strategy: Start with full set, greedily remove using decreasing prime order
    # Then try to add back points via local improvement
    
    # Start with full set [0, N]
    points = list(range(N + 1))
    
    # Phase 1: Greedy removal in decreasing prime order
    points = _greedy_remove(points, primes, deadline)
    
    # Phase 2: Local improvement - try to add back points
    if time.time() < deadline - 10:
        points = _local_improve(points, primes, deadline)
    
    # Final admissibility check
    if not _is_admissible(points, primes):
        # Fallback: simple construction
        pts2 = [1] + [p for p in _primes(N) if p * p > N]
        changed2 = True
        while changed2:
            if time.time() > deadline:
                break
            changed2 = False
            for p in _primes(len(pts2)):
                classes = {}
                for v in pts2:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    pts2 = [v for v in pts2 if v not in smallest]
                    changed2 = True
                    break
        return pts2
    
    return points
# EVOLVE-BLOCK-END
