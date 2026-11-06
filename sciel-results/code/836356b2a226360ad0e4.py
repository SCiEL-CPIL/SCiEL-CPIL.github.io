"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

import time
import math

_START_TIME = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill

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

def _time_left():
    return _START_TIME + _DEADLINE - time.time()

def construct():
    deadline = _START_TIME + _DEADLINE
    primes_cache = {}
    
    def get_primes(limit):
        if limit not in primes_cache:
            primes_cache[limit] = _primes(limit)
        return primes_cache[limit]
    
    # Precompute primes up to N
    all_primes = get_primes(N)
    primes_set = set(all_primes)
    
    # Start with a heuristic set: 0 and all primes > sqrt(N)
    points = set()
    points.add(0)
    for p in all_primes:
        if p * p > N:
            points.add(p)
    
    # Phase 1: Repair initial set - remove points from least-populated residue class
    changed = True
    while changed and _time_left() > 10:
        changed = False
        current_size = len(points)
        for p in get_primes(current_size):
            residue_counts = {}
            for v in points:
                r = v % p
                residue_counts[r] = residue_counts.get(r, 0) + 1
            if len(residue_counts) == p:
                # All residues present, remove from least populated
                min_count = min(residue_counts.values())
                target = min([r for r, c in residue_counts.items() if c == min_count])
                to_remove = [v for v in points if v % p == target]
                removed = min(to_remove)
                points.remove(removed)
                changed = True
                break
    
    # Phase 2: Greedy addition with smart ordering
    # Sort candidates: primes first, then composites by value
    all_candidates = list(range(0, N + 1))
    all_candidates.sort(key=lambda x: (0 if x in primes_set else 1, x))
    
    # For efficiency, maintain residue sets for small primes
    # We'll check admissibility incrementally using precomputed residue info
    
    # Precompute residue membership for efficiency
    # For each prime p up to max possible size, track which residues are present
    
    for cand in all_candidates:
        if _time_left() < 5:
            break
        if cand in points:
            continue
        
        # Check if adding cand keeps set admissible
        new_size = len(points) + 1
        valid = True
        for p in get_primes(new_size):
            # Count residues of current points mod p
            residues = set()
            for v in points:
                residues.add(v % p)
            
            if len(residues) == p:
                # Already all residues covered - violation
                valid = False
                break
            
            # Check if adding cand would complete all p residues
            if (len(residues) == p - 1) and ((cand % p) not in residues):
                valid = False
                break
        
        if valid:
            points.add(cand)
    
    # Phase 3: Local improvement - try swaps to increase size
    # For each point, try replacing it with a candidate that allows adding more
    improved = True
    while improved and _time_left() > 20:
        improved = False
        current_points = list(points)
        for i, old_point in enumerate(current_points):
            if _time_left() < 20:
                break
            # Try removing old_point and adding a different candidate
            for cand in all_candidates:
                if _time_left() < 20:
                    break
                if cand == old_point or cand in points:
                    continue
                
                # Try swap: remove old_point, add cand
                test_points = set(points)
                test_points.remove(old_point)
                test_points.add(cand)
                
                # Check admissibility
                new_size = len(test_points)
                valid = True
                for p in get_primes(new_size):
                    residues = set()
                    for v in test_points:
                        residues.add(v % p)
                    if len(residues) == p:
                        valid = False
                        break
                
                if valid:
                    # Accept swap and try to add more
                    points = test_points
                    improved = True
                    break
            if improved:
                break
    
    return list(points)
