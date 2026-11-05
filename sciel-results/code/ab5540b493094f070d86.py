import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 330.0  # 330s wall-clock deadline, keep margin

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

def _find_violating_prime(points, primes_list):
    """Find a prime p <= len(points) where all residues are covered, or None if admissible."""
    k = len(points)
    if k == 0:
        return None
    
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
                    return p
    return None

def _compute_residue_counts(points, primes_list, max_prime):
    """Compute residue counts for all primes up to max_prime."""
    counts = {}
    for p in primes_list:
        if p > max_prime:
            break
        cnt = [0] * p
        for v in points:
            cnt[v % p] += 1
        counts[p] = cnt
    return counts

def _can_add_candidate_fast(x, missing_residues, primes_list, new_size):
    """
    Fast check if adding x keeps the set admissible.
    missing_residues: dict mapping p -> list of residues with count 0 in current set
    For p <= new_size, we need at least one residue to remain 0 after adding x.
    If missing_residues[p] has >= 2 entries, x is fine (at least one stays 0).
    If missing_residues[p] has exactly 1 entry r0, x % p must != r0.
    If missing_residues[p] is empty, current set isn't admissible (shouldn't happen).
    """
    for p in primes_list:
        if p > new_size:
            break
        mr = missing_residues.get(p, [])
        if len(mr) <= 1:
            if len(mr) == 0:
                return False
            if x % p == mr[0]:
                return False
    return True

def _build_missing_residues(current_set, primes_list, max_prime):
    """Build dict of missing residues for each prime up to max_prime."""
    missing = {}
    for p in primes_list:
        if p > max_prime:
            break
        seen = bytearray(p)
        for v in current_set:
            seen[v % p] = 1
        mr = [i for i in range(p) if not seen[i]]
        missing[p] = mr
    return missing

def construct():
    global START_TIME
    start_time = START_TIME
    
    # Precompute primes up to N (5000)
    primes = _primes(N)
    
    best_set = set()
    best_size = 0
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Start with all numbers
        current_set_set = set(range(N))
        
        # Greedy Repair Loop
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            k = len(current_set_set)
            if k == 0:
                break
            
            # Find a violating prime
            violating_p = None
            for p in primes:
                if p > k:
                    break
                seen = bytearray(p)
                count = 0
                for v in current_set_set:
                    r = v % p
                    if not seen[r]:
                        seen[r] = 1
                        count += 1
                        if count == p:
                            violating_p = p
                            break
                if violating_p:
                    break
            
            if violating_p is None:
                break
            
            # Group elements by residue class mod violating_p
            classes = [[] for _ in range(violating_p)]
            for v in current_set_set:
                classes[v % violating_p].append(v)
            
            # Choose a class to remove. Try different strategies:
            # 1. Smallest class (greedy)
            # 2. Random small class
            # 3. Class that helps most other constraints (heuristic)
            
            class_sizes = [(len(c), i) for i, c in enumerate(classes)]
            class_sizes.sort()
            
            strategy = random.random()
            if strategy < 0.5:
                min_idx = class_sizes[0][1]
            elif strategy < 0.8:
                top_k = min(5, len(class_sizes))
                min_idx = class_sizes[random.randint(0, top_k - 1)][1]
            else:
                # Pick from bottom 10%
                bottom_frac = max(1, len(class_sizes) // 10)
                min_idx = class_sizes[random.randint(0, bottom_frac - 1)][1]
            
            to_remove = set(classes[min_idx])
            current_set_set -= to_remove
        
        # Now current_set is admissible (or empty)
        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
        
        # Add-back phase with efficient checking
        max_add_passes = 20
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            k = len(current_set_set)
            if k == 0:
                break
            
            # Build missing residues for current set (up to k+1 since we might add one)
            missing_residues = _build_missing_residues(current_set_set, primes, k + 1)
            
            candidates = [x for x in range(N) if x not in current_set_set]
            random.shuffle(candidates)
            
            added_any = False
            check_limit = 5000
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 3:
                    break
                
                new_size = k + 1
                if _can_add_candidate_fast(x, missing_residues, primes, new_size):
                    current_set_set.add(x)
                    k += 1
                    added_any = True
                    # Update missing_residues: for each p <= k, if x % p was the only missing, remove it
                    # For p that wasn't in missing_residues (p > old k but <= new k), compute it
                    for p in primes:
                        if p > k:
                            break
                        if p in missing_residues:
                            r = x % p
                            if r in missing_residues[p]:
                                missing_residues[p].remove(r)
                        else:
                            # p was > old k, now p <= k, need to compute missing residues
                            seen = bytearray(p)
                            for v in current_set_set:
                                seen[v % p] = 1
                            missing_residues[p] = [i for i in range(p) if not seen[i]]
            
            if not added_any:
                break
        
        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
    
    return list(best_set)

# Initial call to set start time and warm up
construct()
