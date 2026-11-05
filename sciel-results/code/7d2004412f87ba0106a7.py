import time
import random
import math

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
    
    # Precompute primes up to N (5000)
    primes = _primes(N)
    
    best_set = set()
    best_size = 0
    
    # Strategy: Direct construction using known admissible tuple structure
    # Admissible tuples are related to prime k-tuples that are not "blocked"
    # A common approach is to use the "shifted primes" or known patterns
    
    # The key insight: for a tuple to be admissible, for each prime p <= k,
    # the tuple must not cover all residues mod p.
    # 
    # A direct approach: Start with a large candidate set and iteratively remove
    # the "cheapest" elements to make it admissible.
    #
    # Alternative: Use the structure of admissible tuples. 
    # For k elements, we need to ensure for each prime p <= k, at least one residue is missed.
    #
    # Let's use a greedy repair approach but with a smarter initialization.
    
    # Known good starting point: the set of all numbers 0..N-1 is not admissible
    # because for p=2, it covers both 0 and 1 mod 2.
    # We need to remove elements to break coverage for small primes.
    
    # Direct formula approach:
    # Consider the set S = {x in [0, N) : x mod p != r_p for all primes p <= k}
    # where r_p is a chosen residue. This is the Chinese Remainder Theorem construction.
    # But we want to MAXIMIZE the size, so we want to choose r_p to minimize removals.
    
    # For a given k, the optimal choice is to remove the residue class that appears
    # least frequently in [0, N). For uniform distribution, all classes appear ~N/p times.
    # So removing any class for prime p removes ~N/p elements.
    #
    # But we can be smarter: we only need to ensure admissibility, not remove entire classes.
    # We can remove a minimal set of elements that breaks coverage for all primes p <= k.
    
    # Greedy algorithm:
    # 1. Start with full set [0, N).
    # 2. While not admissible:
    #    a. Find a prime p <= k where all residues are covered.
    #    b. Remove the element in the least common residue class (or random small class) for that p.
    #    c. Repeat.
    
    # To maximize size, we want to remove as few elements as possible.
    # The minimum number of removals is at least max over primes p <= k of (N - N/p) ... no, that's not right.
    # Actually, we need to remove enough elements so that for each p <= k, at least one residue is missing.
    # This is a hitting set / covering problem.
    
    # Let's implement a fast greedy repair with randomization for diversity.
    
    num_restarts = 500  # More restarts for better exploration
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Start with all numbers
        current_set = list(range(N))
        current_set_set = set(current_set)
        
        # Greedy Repair Loop
        # We want to make it admissible by removing elements.
        # To maximize size, we remove the "cheapest" constraint violation.
        
        max_iterations = 10000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            k = len(current_set)
            if k == 0:
                break
            
            # Find a violating prime
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
            
            # We need to remove elements to break the coverage for violating_p
            # Group elements by residue class mod violating_p
            classes = [[] for _ in range(violating_p)]
            for v in current_set:
                classes[v % violating_p].append(v)
            
            # Find the class with the minimum number of elements
            # Removing this class minimizes the loss in size
            class_sizes = [(len(c), i) for i, c in enumerate(classes)]
            class_sizes.sort()
            
            # Randomly pick from the top few smallest classes for diversity
            if random.random() < 0.8:
                min_idx = class_sizes[0][1]
            else:
                top_k = min(5, len(class_sizes))
                min_idx = class_sizes[random.randint(0, top_k - 1)][1]
            
            # Remove all elements in this class
            to_remove = set(classes[min_idx])
            current_set_set -= to_remove
            # Rebuild current_set as a list for efficient iteration
            current_set = list(current_set_set)
        
        # Now current_set is admissible (or empty)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
        # Try to add back elements to improve size
        # This is a local search: try adding any element from [0, N) not in current_set
        # If it remains admissible, keep it.
        
        # To be efficient, we maintain residue counts for current_set for all relevant primes.
        # But for simplicity and speed, we do a few passes of random additions.
        
        max_add_passes = 10
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
                
            # Get candidates: elements not in current_set
            candidates = [x for x in range(N) if x not in current_set_set]
            random.shuffle(candidates)
            
            # Limit candidates to check to avoid timeout
            check_limit = 1000
            added_any = False
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 2:
                    break
                
                # Check if current_set U {x} is admissible
                new_size = len(current_set_set) + 1
                is_adm = True
                for p in primes:
                    if p > new_size:
                        break
                    # Check if all residues mod p are covered in current_set U {x}
                    seen = bytearray(p)
                    count = 0
                    for v in current_set_set:
                        r = v % p
                        if not seen[r]:
                            seen[r] = 1
                            count += 1
                            if count == p:
                                # This shouldn't happen if current_set is admissible
                                break
                    rx = x % p
                    if not seen[rx]:
                        count += 1
                        if count == p:
                            is_adm = False
                            break
                if is_adm:
                    current_set_set.add(x)
                    added_any = True
            
            # If we added none, we are likely at a local maximum
            if not added_any:
                break
        
        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
            
    return list(best_set)

# Initial call to set start time and warm up
construct()
