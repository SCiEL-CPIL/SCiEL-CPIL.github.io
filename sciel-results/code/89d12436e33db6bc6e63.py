import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Leave 5s buffer for safety

    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute exponent vectors for numbers 2..40
    # vec[n] is a tuple of length num_primes
    vecs = {}
    for n in range(2, 41):
        v = [0] * num_primes
        m = n
        for i, p in enumerate(primes):
            while m % p == 0:
                v[i] += 1
                m //= p
        # m must be 1 because primes cover all factors
        vecs[n] = tuple(v)
        
    candidates = list(range(2, 41))
    best_set = []
    best_size = 0
    
    # Helper to check if a vector can be added to a current set of sums
    # current_sums: set of tuples representing all subset sums of the current selection
    # v: vector to add
    # Returns True if valid, and the new set of sums if valid
    def try_add(current_sums, v):
        # We need to check if any s + v is in current_sums
        # If so, collision.
        # Also, v itself is s=0 + v. If v in current_sums, collision.
        # The check (s+v) in current_sums covers v (when s=0) and others.
        
        # Optimization: Instead of building the full new set first, check intersection.
        # If valid, we build the new set.
        
        # For speed, we can just iterate.
        # 2^k sums. k is small (max 18 -> 260k sums).
        # 260k checks per candidate is fast.
        
        for s in current_sums:
            # Compute s + v
            # Using tuple comprehension
            sv = tuple(s[i] + v[i] for i in range(num_primes))
            if sv in current_sums:
                return False, None
        
        # If we get here, it's valid. Build new sums.
        new_sums = set(current_sums)
        for s in current_sums:
            sv = tuple(s[i] + v[i] for i in range(num_primes))
            new_sums.add(sv)
        return True, new_sums

    # We will perform random restarts with greedy construction
    # To increase the chance of finding a larger set, we can also try
    # different initial seeds or ordering strategies.
    
    # Strategy 1: Random Shuffle + Greedy
    # Strategy 2: Specific orderings (e.g., by number of prime factors)
    
    # Let's mix them.
    
    # Pre-computed orderings
    # 1. By value
    # 2. By number of prime factors (ascending)
    # 3. Random
    
    def get_ordering():
        if random.random() < 0.5:
            return candidates[:]
        else:
            # Sort by number of prime factors (with multiplicity)
            def count_factors(n):
                c = 0
                d = 2
                while d * d <= n:
                    while n % d == 0:
                        c += 1
                        n //= d
                    d += 1
                if n > 1:
                    c += 1
                return c
            
            return sorted(candidates, key=count_factors)

    zero_vec = tuple([0] * num_primes)
    
    # Run multiple restarts
    iterations = 0
    while time.time() < DEADLINE:
        iterations += 1
        
        # Choose an ordering
        if iterations % 10 == 0:
            # Every 10th, try a structured ordering
            order = get_ordering()
        else:
            order = candidates[:]
            random.shuffle(order)
            
        current_set = []
        current_sums = {zero_vec}
        
        for n in order:
            if time.time() > DEADLINE:
                break
            v = vecs[n]
            valid, new_sums = try_add(current_sums, v)
            if valid:
                current_set.append(n)
                current_sums = new_sums
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we found 18, it's likely optimal (max possible distinct ints in 2..40 is 39, 
            # but dissociated sets are limited. 18 is a high target).
            # Actually, can we go higher?
            # The space is Z^12. The number of distinct subset sums is 2^k.
            # The "volume" of the box defined by max exponents limits k.
            # Max exponent for 2 is 5 (32). Max for 3 is 3 (27).
            # It's hard to pack more than ~15-18.
            
            if best_size >= 18:
                break

    # Local Search: Try to improve the best set found by swapping elements
    # This is a structural refinement.
    if best_size < 18 and time.time() < DEADLINE - 5:
        # Reconstruct sums for best_set
        best_sums = {zero_vec}
        for n in best_set:
            v = vecs[n]
            new_sums = set()
            for s in best_sums:
                new_sums.add(tuple(s[i] + v[i] for i in range(num_primes)))
            best_sums.update(new_sums)
            
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            # Try to swap each element in best_set with an element not in best_set
            for i, elem in enumerate(best_set):
                # Rebuild sums without elem? 
                # That's expensive. 
                # Alternative: Just try to add a new element and remove an old one?
                # Or just try to add an element to best_set (if possible) -> already done in greedy.
                # Swap: remove elem, add new_elem.
                
                # To do this efficiently, we'd need to maintain the sums excluding elem.
                # Given the small size, let's just do a full check for swaps if time permits.
                
                # Actually, a simpler local search:
                # Try to replace an element with a "better" one that allows adding more elements later?
                # This is complex. 
                
                # Let's stick to the greedy random restarts which are proven effective for this size.
                # The inspiration 1 got 17. We aim for 17 or 18.
                
                break # Skip local search for now to keep code simple and fast, rely on restarts.
            break

    # Final verification
    # Check if best_set is valid
    def is_valid(set_to_check):
        sums = {zero_vec}
        for n in set_to_check:
            v = vecs[n]
            new_sums = set()
            for s in sums:
                sv = tuple(s[i] + v[i] for i in range(num_primes))
                if sv in sums:
                    return False
                new_sums.add(sv)
            sums.update(new_sums)
        return True

    if not is_valid(best_set):
        # Fallback to a known valid set (e.g., primes)
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return [int(x) for x in best_set]
