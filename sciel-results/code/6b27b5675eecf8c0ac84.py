import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s deadline, keep margin

# Precompute primes up to N
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

PRECOMPUTED_PRIMES = _primes(N)

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

def _min_residue_class(points, p):
    """Find the residue class mod p with the fewest elements."""
    counts = [0] * p
    for v in points:
        counts[v % p] += 1
    min_count = min(counts)
    min_residues = [r for r in range(p) if counts[r] == min_count]
    return min_residues, min_count

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    primes = PRECOMPUTED_PRIMES
    best_set = set()
    best_size = 0
    
    # Strategy: Model as maximum independent set / vertex cover problem
    # For each prime p <= k, we need at least one residue class to be missed.
    # This is equivalent to: for each p, not all residues are present.
    # 
    # Key insight: We can frame this as a hitting set problem where we want to
    # remove the minimum number of elements such that for each prime p <= k,
    # at least one residue class mod p has zero elements remaining.
    #
    # Greedy approach: Start with full set, iteratively remove the "cheapest"
    # residue class for a violating prime.
    #
    # Improvement: Use a smarter initialization and more aggressive local search.
    
    num_restarts = 200  # Reduced restarts, more time per restart for better optimization
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        # Start with all numbers
        current_set = set(range(N))
        
        # Greedy Repair Loop with optimized residue tracking
        # Maintain residue counts for fast violation detection
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            k = len(current_set)
            if k == 0:
                break
            
            # Find a violating prime efficiently
            violating_p = None
            for p in primes:
                if p > k:
                    break
                # Check if all residues mod p are covered
                seen = bytearray(p)
                count = 0
                for v in current_set:
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
                # Admissible
                break
            
            # Group elements by residue class mod violating_p
            classes = [[] for _ in range(violating_p)]
            for v in current_set:
                classes[v % violating_p].append(v)
            
            # Find the class with the minimum number of elements
            # Removing this class minimizes the loss in size
            class_sizes = [(len(c), i) for i, c in enumerate(classes)]
            class_sizes.sort()
            
            # Randomly pick from the top few smallest classes for diversity
            if random.random() < 0.7:
                min_idx = class_sizes[0][1]
            else:
                top_k = min(3, len(class_sizes))
                min_idx = class_sizes[random.randint(0, top_k - 1)][1]
            
            # Remove all elements in this class
            to_remove = set(classes[min_idx])
            current_set -= to_remove
        
        # Now current_set is admissible (or empty)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        
        # Local search: try to add back elements to improve size
        # Use a more efficient check by precomputing residue presence
        
        max_add_passes = 20
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Get candidates: elements not in current_set
            candidates = [x for x in range(N) if x not in current_set]
            random.shuffle(candidates)
            
            # Limit candidates to check to avoid timeout
            check_limit = 2000
            added_any = False
            
            # Precompute current residue presence for each prime
            # This is expensive, so we do it once per pass
            residue_presence = {}
            for p in primes:
                if p > len(current_set) + check_limit:
                    continue
                presence = bytearray(p)
                for v in current_set:
                    presence[v % p] = 1
                residue_presence[p] = presence
            
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 3:
                    break
                
                # Check if current_set U {x} is admissible
                new_size = len(current_set) + 1
                is_adm = True
                for p in primes:
                    if p > new_size:
                        break
                    if p not in residue_presence:
                        # Compute on the fly
                        presence = bytearray(p)
                        for v in current_set:
                            presence[v % p] = 1
                        residue_presence[p] = presence
                    else:
                        presence = residue_presence[p]
                    
                    rx = x % p
                    if not presence[rx]:
                        # Adding x would complete this residue class
                        # Check if this would make all residues present
                        count = sum(presence) + 1
                        if count == p:
                            is_adm = False
                            break
                    else:
                        # x's residue is already present, no change in coverage
                        pass
                
                if is_adm:
                    current_set.add(x)
                    added_any = True
                    # Update residue presence for added x
                    for p in primes:
                        if p > len(current_set):
                            break
                        if p in residue_presence:
                            residue_presence[p][x % p] = 1
                        else:
                            presence = bytearray(p)
                            for v in current_set:
                                presence[v % p] = 1
                            residue_presence[p] = presence
            
            # If we added none, we are likely at a local maximum
            if not added_any:
                break
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
    return list(best_set)

# Initial call to set start time and warm up
construct()
