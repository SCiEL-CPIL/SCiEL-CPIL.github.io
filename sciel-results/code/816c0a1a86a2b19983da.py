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

def _can_add_candidate(x, current_set, residue_counts, primes_list, new_size):
    """Check if adding x to current_set keeps it admissible using precomputed counts."""
    for p in primes_list:
        if p > new_size:
            break
        cnt = residue_counts[p]
        r = x % p
        # If this residue class already has count >= p-1 and we'd make it p, 
        # but actually we need to check if ALL residues would be covered
        # A residue class is "full" if count == p
        # After adding x, residue r gets count+1
        # The set is NOT admissible if ALL p residue classes have count >= 1 after adding
        # Since current_set is admissible, at most one residue class can have count == p-1
        # Actually, let's think differently: current_set is admissible means for each p<=len(current_set),
        # not all residues are covered. When we add x, new_size = len+1, so we need to check
        # primes p <= new_size.
        # For p <= len(current_set): current_set is admissible, so some residue is 0.
        #   After adding x, that residue stays 0 unless x falls in it, but even then it becomes 1,
        #   still not all covered (since at least one was 0 and now might be 1, but others could be 0 too)
        #   Wait, no. If current_set has exactly one missing residue, and x fills it, then all are covered.
        # So we need: for each p <= new_size, after adding x, not all residues are covered.
        
        # Efficient check: count how many residues have count == 0 in current_set
        # If zero residues have count 0, current_set isn't admissible (contradiction)
        # If exactly one residue r0 has count 0, then x must NOT have x % p == r0
        # If two or more residues have count 0, x can go anywhere
        
        zeros = 0
        for i in range(p):
            if cnt[i] == 0:
                zeros += 1
                if zeros > 1:
                    break
        
        if zeros == 0:
            # Shouldn't happen if current_set is admissible
            return False
        elif zeros == 1:
            # Find the missing residue
            for i in range(p):
                if cnt[i] == 0:
                    if r == i:
                        return False
                    break
        # else: zeros >= 2, any x is fine for this prime
    return True

def construct():
    global START_TIME
    start_time = START_TIME
    
    # Precompute primes up to N (5000)
    primes = _primes(N)
    
    best_set = set()
    best_size = 0
    
    num_restarts = 150
    
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
        max_add_passes = 10
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            k = len(current_set_set)
            if k == 0:
                break
            
            # Precompute residue counts for current set
            residue_counts = {}
            for p in primes:
                if p > k + 1:  # We'll check up to new_size = k+1
                    break
                cnt = [0] * p
                for v in current_set_set:
                    cnt[v % p] += 1
                residue_counts[p] = cnt
            
            candidates = [x for x in range(N) if x not in current_set_set]
            random.shuffle(candidates)
            
            added_any = False
            check_limit = 2000
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 3:
                    break
                
                new_size = k + 1
                if _can_add_candidate(x, current_set_set, residue_counts, primes, new_size):
                    current_set_set.add(x)
                    k += 1
                    added_any = True
                    # Update residue counts
                    for p in primes:
                        if p > k:
                            break
                        if p in residue_counts:
                            residue_counts[p][x % p] += 1
                        else:
                            # Need to add this prime
                            cnt = [0] * p
                            for v in current_set_set:
                                cnt[v % p] += 1
                            residue_counts[p] = cnt
            
            if not added_any:
                break
        
        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
    
    return list(best_set)

# Initial call to set start time and warm up
construct()
