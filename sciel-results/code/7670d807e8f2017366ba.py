import time
import random
import math
import numpy as np

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s deadline, keep margin

def _primes(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _is_admissible(points, primes_list):
    """
    Check if the set of points is admissible.
    Admissible: for every prime p <= len(points), some residue class mod p is missed.
    """
    k = len(points)
    if k == 0:
        return True
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return False
    return True

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    # Precompute primes up to N (5000)
    primes = _primes(N)
    
    # Precompute residue vectors for all primes for fast counting
    prime_residues = {}
    for p in primes:
        if p <= N:
            prime_residues[p] = np.arange(N) % p
    
    best_set = set()
    best_size = 0
    
    # Strategy: Greedy removal + fast vectorized add-back
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 25:
            break
        
        # Start with all numbers
        in_set = np.ones(N, dtype=bool)
        
        # Removal phase: iteratively remove least-populated residue class from violating primes
        max_iterations = 4000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
                
            current_size = int(np.sum(in_set))
            if current_size == 0:
                break
            
            # Find first violating prime
            violating_p = None
            for p in primes:
                if p > current_size:
                    break
                
                residues = prime_residues[p]
                mask = in_set
                # Check if all residues are covered using vectorized ops
                # Create a boolean array of size p, mark covered residues
                covered = np.zeros(p, dtype=bool)
                # For each residue r, check if any element in set has that residue
                for r in range(p):
                    if np.any(residues[mask] == r):
                        covered[r] = True
                
                if np.all(covered):
                    violating_p = p
                    break
            
            if violating_p is None:
                # Admissible
                break
            
            # Find the residue class with the fewest elements in the violating prime
            residues = prime_residues[violating_p]
            mask = in_set
            counts = np.zeros(violating_p, dtype=np.int64)
            for r in range(violating_p):
                counts[r] = np.sum(residues[mask] == r)
            
            # Pick the minimum count residue class (randomly if ties)
            min_count = int(np.min(counts))
            min_residues = np.where(counts == min_count)[0]
            min_idx = int(random.choice(min_residues))
            
            # Remove all elements in this residue class
            to_remove = (residues == min_idx) & mask
            in_set[to_remove] = False
        
        current_set = set(np.where(in_set)[0].tolist())
        current_size = len(current_set)
        
        if current_size > best_size:
            best_set = set(current_set)
            best_size = current_size
        
        # Fast add-back phase with precomputed residue coverage
        max_add_passes = 8
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 8:
                break
            
            current_size = len(current_set)
            if current_size == 0:
                break
            
            # Precompute residue coverage for current_set for all active primes
            # active_primes: p <= current_size (since we're checking if adding one keeps it admissible,
            # we need p <= current_size + 1, but p must be <= new_size for the constraint to apply)
            active_primes = [p for p in primes if p <= current_size + 1]
            
            # For each active prime, precompute which residues are covered and how many elements per residue
            residue_info = {}
            for p in active_primes:
                res_p = prime_residues[p]
                # Get residues of all elements in current_set
                # Use list comprehension for speed
                elems = list(current_set)
                residues_arr = np.array([v % p for v in elems], dtype=np.int64)
                counts_arr = np.zeros(p, dtype=np.int64)
                for r in range(p):
                    counts_arr[r] = np.sum(residues_arr == r)
                residue_info[p] = counts_arr
            
            # Get candidates
            candidates = np.array([x for x in range(N) if x not in current_set], dtype=np.int64)
            if len(candidates) == 0:
                break
            
            # Limit candidates per pass
            check_limit = min(len(candidates), 1000)
            random_indices = np.random.choice(len(candidates), size=check_limit, replace=False)
            candidates = candidates[random_indices]
            
            added_any = False
            
            for x in candidates:
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                x = int(x)
                new_size = current_size + 1
                
                # Check if adding x maintains admissibility
                is_adm = True
                for p in active_primes:
                    if p > new_size:
                        break
                    
                    counts_arr = residue_info[p]
                    rx = x % p
                    
                    # If this residue class is already empty (count 0), adding x makes it count 1
                    # If count was p-1, adding x makes it p (violation!)
                    # If count was < p-1, adding x keeps it < p (OK)
                    # If count was p, it was already a violation (shouldn't happen if we're admissible)
                    
                    if counts_arr[rx] == p - 1:
                        # Adding x would complete all residues mod p -> violation
                        is_adm = False
                        break
                    # else: counts_arr[rx] < p-1, so after adding, count <= p-1, still missed
                    # Note: counts_arr[rx] can't be p if we're currently admissible
                
                if is_adm:
                    current_set.add(x)
                    added_any = True
                    current_size += 1
                    # Update residue_info for the new element
                    for p in active_primes:
                        if p <= current_size + 1:
                            rx = x % p
                            residue_info[p][rx] += 1
                    # Add new prime to active_primes if needed
                    for p in primes:
                        if p > current_size + 1:
                            break
                        if p not in residue_info:
                            # Compute residue info for this new prime
                            elems = list(current_set)
                            residues_arr = np.array([v % p for v in elems], dtype=np.int64)
                            counts_arr = np.zeros(p, dtype=np.int64)
                            for r in range(p):
                                counts_arr[r] = np.sum(residues_arr == r)
                            residue_info[p] = counts_arr
                    active_primes = [p for p in primes if p <= current_size + 1]
            
            if not added_any:
                break
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
    return list(best_set)

# Initial call to set start time and warm up
construct()
