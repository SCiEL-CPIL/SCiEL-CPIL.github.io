"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    """Sieve of Eratosthenes to find all primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _greedy_construct(all_primes, seed=None):
    """
    Constructs an admissible tuple using a greedy strategy with optional randomization.
    """
    if seed is not None:
        random.seed(seed)
        
    points = list(range(N + 1))
    current_len = len(points)
    
    # We need to process primes p <= current_len.
    # As points are removed, current_len decreases.
    # We iterate through primes in increasing order.
    
    for p in all_primes:
        if p > current_len:
            break
            
        # Count residues
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
            
        # Find minimum count
        min_count = min(counts)
        
        # Find all classes with min count
        min_classes = [i for i, c in enumerate(counts) if c == min_count]
        
        # Choose a class to remove
        if len(min_classes) == 1:
            target_class = min_classes[0]
        else:
            target_class = random.choice(min_classes)
            
        # Remove points in target_class
        points = [v for v in points if v % p != target_class]
        current_len = len(points)
        
    return points

def _local_improvement(points, all_primes):
    """
    Attempts to add elements back to the set to maximize size.
    """
    points_set = set(points)
    current_len = len(points_set)
    
    # Precompute missing residues for all primes p <= current_len
    # missing_residues[p] = r such that no x in points has x % p == r
    missing_residues = {}
    
    def update_missing_residues():
        nonlocal current_len
        current_len = len(points_set)
        missing_residues = {}
        for p in all_primes:
            if p > current_len:
                break
            counts = [0] * p
            for v in points_set:
                counts[v % p] += 1
            # Find a missing residue
            for r in range(p):
                if counts[r] == 0:
                    missing_residues[p] = r
                    break
            else:
                # Should not happen if admissible
                return False
        return True

    if not update_missing_residues():
        return points_set

    improved = True
    while improved:
        improved = False
        current_len = len(points_set)
        
        # Check candidates
        # We can try to add any x in [0, N] not in points_set
        # To be efficient, we can iterate through a shuffled list of candidates
        candidates = [x for x in range(N + 1) if x not in points_set]
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x is safe to add
            safe = True
            
            # 1. For all p <= current_len, x must not be the missing residue
            for p, r in missing_residues.items():
                if x % p == r:
                    safe = False
                    break
            
            if not safe:
                continue
                
            # 2. Check the new prime p_new = current_len + 1
            p_new = current_len + 1
            # p_new might not be prime, but the condition applies to primes p <= len(tuple).
            # If p_new is not prime, we don't need to check it specifically for "new" constraint?
            # The condition is: for every prime p <= len(tuple).
            # If we add x, new len is current_len + 1.
            # We need to check primes p <= current_len + 1.
            # We already checked primes p <= current_len.
            # We only need to check if there is a prime p such that current_len < p <= current_len + 1.
            # This is only possible if current_len + 1 is prime.
            if p_new in [p for p in all_primes if p <= N+1]: # Optimization: check if prime
                # We need to ensure S U {x} misses a residue mod p_new.
                # S has size p_new - 1.
                # S misses exactly one residue r_new mod p_new.
                # If x % p_new == r_new, then S U {x} covers all residues. Bad.
                # If x % p_new != r_new, then S U {x} still misses r_new. Good.
                
                # Find r_new
                counts_new = [0] * p_new
                for v in points_set:
                    counts_new[v % p_new] += 1
                
                r_new = -1
                for r in range(p_new):
                    if counts_new[r] == 0:
                        r_new = r
                        break
                
                if r_new != -1 and x % p_new == r_new:
                    safe = False
            
            if safe:
                points_set.add(x)
                improved = True
                break # Restart loop as set changed

    return list(points_set)

def construct():
    deadline = time.time() + 280  # safety margin
    all_primes = _primes(N + 1)
    
    best_points = []
    best_size = -1
    
    num_runs = 0
    # Run randomized greedy multiple times
    while time.time() < deadline - 20: # Leave 20s for local improvement
        start_time = time.time()
        points = _greedy_construct(all_primes, seed=num_runs)
        size = len(points)
        
        if size > best_size:
            best_size = size
            best_points = points
            
        num_runs += 1
        
        # If we've done enough runs or time is tight, break
        if num_runs > 2000:
            break

    # Apply local improvement to the best point set
    if time.time() < deadline - 5:
        improved_points = _local_improvement(best_points, all_primes)
        if len(improved_points) > best_size:
            best_points = improved_points

    return best_points
# EVOLVE-BLOCK-END
