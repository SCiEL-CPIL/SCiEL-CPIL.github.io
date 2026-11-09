import random
import time

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def check_valid(vals):
    """
    vals: sorted list of unique integers
    returns True if all consecutive block products are distinct
    """
    prods = set()
    n = len(vals)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= vals[j]
            if prod in prods:
                return False
            prods.add(prod)
    return True

def construct():
    # Pre-compute primes and composites
    numbers = list(range(2, 129))
    primes = [x for x in numbers if is_prime(x)]
    composites = [x for x in numbers if not is_prime(x)]
    
    # Base solution: all primes
    # Primes are guaranteed to form a valid sequence because
    # any block product is a unique combination of primes (Fundamental Theorem of Arithmetic)
    # and since the sequence is strictly increasing, no two blocks can have the same set of primes.
    best_set = set(primes)
    best_len = len(best_set)
    
    start_time = time.time()
    iterations = 0
    
    # Strategy 1: Greedy insertion of composites
    # We try to add composites to the prime base.
    # The order of addition might matter, so we use randomized restarts.
    
    # Optimization: Sort composites by "risk" or just randomize.
    # Composites with small factors are riskier.
    
    while time.time() - start_time < 50:
        iterations += 1
        current_set = set(primes)
        
        # Randomize the order of composites to explore different paths
        shuffled_composites = composites[:]
        random.shuffle(shuffled_composites)
        
        for c in shuffled_composites:
            # Skip if already in set (shouldn't happen as primes and composites are disjoint)
            if c in current_set:
                continue
            
            new_set = current_set | {c}
            new_vals = sorted(new_set)
            
            # Check validity
            if check_valid(new_vals):
                current_set = new_set
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set
            
            # If we found a new best, we can continue searching to see if we can add more
            # or restart from this new state?
            # For now, just record it.
            
        # Early termination if we found a very long sequence?
        # Max possible is 127. Unlikely.
        # If best_len is close to max possible composites + primes, we can stop.
        
    return sorted(best_set)
