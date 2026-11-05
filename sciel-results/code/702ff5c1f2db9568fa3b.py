import time
import random
import math
from collections import defaultdict

N = 5000
START_TIME = time.time()
TIME_LIMIT = 285.0  # 300s deadline, keep margin

# Precompute primes up to N
def _sieve_primes(limit):
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

PRECOMPUTED_PRIMES = _sieve_primes(N)
# Only need primes up to N for checking admissibility of tuples of size <= N
PRIMES_FOR_CHECK = [p for p in PRECOMPUTED_PRIMES if p <= N]

def _is_admissible_fast(residue_counts, primes_list, k):
    """
    Check admissibility using precomputed residue counts.
    residue_counts: dict mapping prime p -> list of counts for each residue 0..p-1
    """
    if k == 0:
        return True
    for p in primes_list:
        if p > k:
            break
        counts = residue_counts.get(p)
        if counts is None:
            return True
        if all(c > 0 for c in counts):
            return False
    return True

def _find_violating_prime_fast(residue_counts, primes_list, k):
    """Find violating prime using precomputed counts."""
    if k == 0:
        return None
    for p in primes_list:
        if p > k:
            break
        counts = residue_counts.get(p)
        if counts is None:
            continue
        if all(c > 0 for c in counts):
            return p
    return None

def _min_residue_class_fast(residue_counts, p):
    """Find residue class with min count for prime p."""
    counts = residue_counts.get(p)
    if counts is None:
        return 0, 0
    min_count = min(counts)
    for r in range(p):
        if counts[r] == min_count:
            return r, min_count
    return 0, 0

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    primes = PRIMES_FOR_CHECK
    best_set = set()
    best_size = 0
    
    # Precompute initial residue counts for the full set [0, N)
    # For each prime p, count residues in [0, N)
    full_residue_counts = {}
    for p in primes:
        if p > N:
            break
        counts = [0] * p
        for v in range(N):
            counts[v % p] += 1
        full_residue_counts[p] = counts
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Start with all numbers
        current_set = set(range(N))
        k = N
        
        # Copy initial residue counts
        residue_counts = {p: list(counts) for p, counts in full_residue_counts.items()}
        
        # Greedy Repair Loop
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
                
            k = len(current_set)
            if k == 0:
                break
            
            # Find a violating prime efficiently using precomputed counts
            violating_p = _find_violating_prime_fast(residue_counts, primes, k)
            
            if violating_p is None:
                break
            
            # Find the residue class with minimum count
            min_res, min_count = _min_residue_class_fast(residue_counts, violating_p)
            
            # With some probability, pick a random small class for diversity
            if random.random() < 0.15 and min_count > 0:
                candidate_residues = [r for r in range(violating_p) 
                                      if residue_counts[violating_p][r] <= min_count + 2]
                if candidate_residues:
                    min_res = random.choice(candidate_residues)
            
            # Remove all elements with this residue class mod violating_p
            to_remove = [v for v in current_set if v % violating_p == min_res]
            
            # Update residue counts for all primes
            for v in to_remove:
                for p in primes:
                    if p > k:
                        break
                    if p in residue_counts:
                        residue_counts[p][v % p] -= 1
            
            for v in to_remove:
                current_set.discard(v)
            k = len(current_set)
        
        # Now current_set is admissible (or empty)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        
        # Local search: try to add back elements to improve size
        max_add_passes = 20
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            k = len(current_set)
            
            # Get candidates: elements not in current_set
            candidates = [x for x in range(N) if x not in current_set]
            random.shuffle(candidates)
            
            check_limit = min(2000, len(candidates))
            added_any = False
            
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                new_size = k + 1
                is_adm = True
                
                for p in primes:
                    if p > new_size:
                        break
                    if p not in residue_counts:
                        continue
                    
                    counts = residue_counts[p]
                    rx = x % p
                    
                    if counts[rx] == 0:
                        # Check if this is the last missing residue
                        missing_count = 0
                        for c in counts:
                            if c == 0:
                                missing_count += 1
                                if missing_count > 1:
                                    break
                        if missing_count == 1:
                            is_adm = False
                            break
                    
                if is_adm:
                    current_set.add(x)
                    added_any = True
                    k += 1
                    for p in primes:
                        if p > k:
                            break
                        if p in residue_counts:
                            residue_counts[p][x % p] += 1
                        else:
                            # Need to add this prime to tracking
                            new_counts = [0] * p
                            for v in current_set:
                                new_counts[v % p] += 1
                            residue_counts[p] = new_counts
        
            if not added_any:
                break
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
    # Final polish: try to add more elements
    if best_size > 0 and time.time() - start_time < TIME_LIMIT - 5:
        final_set = set(best_set)
        k = len(final_set)
        
        # Build residue counts for final_set
        residue_counts = {}
        for p in primes:
            if p > k:
                break
            counts = [0] * p
            for v in final_set:
                counts[v % p] += 1
            residue_counts[p] = counts
        
        candidates = [x for x in range(N) if x not in final_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            
            new_size = k + 1
            is_adm = True
            
            for p in primes:
                if p > new_size:
                    break
                if p not in residue_counts:
                    continue
                
                counts = residue_counts[p]
                rx = x % p
                
                if counts[rx] == 0:
                    missing_count = 0
                    for c in counts:
                        if c == 0:
                            missing_count += 1
                            if missing_count > 1:
                                break
                    if missing_count == 1:
                        is_adm = False
                        break
            
            if is_adm:
                final_set.add(x)
                k += 1
                for p in primes:
                    if p > k:
                        break
                    if p in residue_counts:
                        residue_counts[p][x % p] += 1
        
        if len(final_set) > best_size:
            best_set = final_set
            best_size = len(final_set)
    
    return list(best_set)


