"""Construct the largest admissible tuple within [0, 5000]."""
import time
import random
import numpy as np
from itertools import product

N = 5000
DEADLINE = 330.0
SAFETY_MARGIN = 5.0

def _primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i, v in enumerate(sieve) if v]

def _is_admissible(points, primes):
    """Check if points form an admissible tuple."""
    n = len(points)
    if n == 0:
        return True
    point_arr = np.array(points)
    for p in primes:
        if p > n:
            break
        residues = point_arr % p
        if len(np.unique(residues)) == p:
            return False
    return True

def _greedy_init_vectorized(primes, branch_assignments, start_time, max_time):
    """
    Vectorized greedy initialization given branch assignments.
    branch_assignments: dict prime -> missed residue class
    """
    candidates = np.arange(N + 1)
    
    # Mark incompatible candidates (those in the missed residue class for branched primes)
    compatible = np.ones(N + 1, dtype=bool)
    for p, r in branch_assignments.items():
        compatible[candidates % p == r] = False
    
    # Get compatible candidates
    comp_candidates = candidates[compatible]
    
    # Greedily add points, checking admissibility
    points = []
    point_set = set()
    
    # Precompute primes that matter (p <= N+1)
    relevant_primes = [p for p in primes if p <= N + 1]
    
    for v in comp_candidates:
        if time.time() - start_time > max_time:
            break
        v = int(v)
        new_len = len(points) + 1
        can_add = True
        for p in relevant_primes:
            if p > new_len:
                break
            # Check if adding v would cover all p residue classes
            # Current unique residues mod p
            if points:
                current_residues = set(pt % p for pt in points)
            else:
                current_residues = set()
            if v % p not in current_residues and len(current_residues) == p - 1:
                can_add = False
                break
        if can_add:
            points.append(v)
            point_set.add(v)
    
    return points, point_set

def _hill_climb_vectorized(points, primes, start_time, max_time):
    """
    Hill climb: try to add points to increase tuple size.
    """
    point_set = set(points)
    
    # Try to add points in order
    for v in range(N + 1):
        if time.time() - start_time > max_time:
            break
        if v in point_set:
            continue
        new_len = len(points) + 1
        can_add = True
        for p in primes:
            if p > new_len:
                break
            current_residues = set(pt % p for pt in points)
            if v % p not in current_residues and len(current_residues) == p - 1:
                can_add = False
                break
        if can_add:
            points.append(v)
            point_set.add(v)
    
    return points, point_set

def _branch_first_primes(primes, branch_depth, start_time, max_time):
    """
    Branch over residue class assignments for first `branch_depth` primes.
    """
    best_points = []
    best_score = 0
    
    branched_primes = primes[:branch_depth]
    class_ranges = [range(p) for p in branched_primes]
    
    num_combinations = 1
    for p in branched_primes:
        num_combinations *= p
    
    if num_combinations > 30000:
        rng = random.Random(42)
        samples = set()
        for _ in range(30000):
            assignment = tuple(rng.randint(0, p-1) for p in branched_primes)
            samples.add(assignment)
        assignments_to_try = list(samples)
    else:
        assignments_to_try = list(product(*class_ranges))
    
    random.shuffle(assignments_to_try)
    
    for assignment in assignments_to_try:
        if time.time() - start_time > max_time:
            break
        
        branch_assignments = {}
        for i, p in enumerate(branched_primes):
            branch_assignments[p] = assignment[i]
        
        points, point_set = _greedy_init_vectorized(primes, branch_assignments, start_time, max_time)
        
        if time.time() - start_time > max_time:
            break
        
        points, point_set = _hill_climb_vectorized(points, primes, start_time, max_time)
        
        if len(points) > best_score:
            best_score = len(points)
            best_points = list(points)
    
    return best_points

def construct():
    start_time = time.time()
    max_time = DEADLINE - SAFETY_MARGIN
    
    primes = _primes(N + 1)
    
    # Try increasing branch depths
    best_points = []
    
    for depth in [5, 6]:
        if time.time() - start_time > max_time:
            break
        points = _branch_first_primes(primes, depth, start_time, max_time)
        if len(points) > len(best_points):
            best_points = points
    
    if not best_points:
        sqrt_n = int(N ** 0.5) + 1
        best_points = [1] + [p for p in primes if p * p > N and p <= N]
    
    if not _is_admissible(best_points, primes):
        sqrt_n = int(N ** 0.5) + 1
        best_points = [1] + [p for p in primes if p * p > N and p <= N]
    
    return best_points
