import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # Candidates are integers from 2 to 40
    candidates = list(range(2, 41))
    n_candidates = len(candidates)
    
    # Precompute prime exponents for each candidate
    # Primes up to 40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def prime_exponent_vector(n):
        """Return tuple of prime exponents for n."""
        vec = []
        for p in primes:
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            vec.append(count)
        return tuple(vec)
    
    # Precompute exponent vectors
    exp_vectors = {x: prime_exponent_vector(x) for x in candidates}
    
    best_set = []
    best_size = 0
    
    def can_add(prods, x):
        """Check if x can be added given current subset products prods."""
        new_prods = set()
        for p in prods:
            new_prods.add(x * p)
        # Check no duplicates within new_prods
        if len(new_prods) != len(prods):
            return None
        # Check disjointness with existing prods
        if new_prods & prods:
            return None
        return new_prods
    
    # Try multiple random orderings with greedy addition
    random.seed(42)
    
    num_trials = 2000
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        order = candidates[:]
        random.shuffle(order)
        
        current_set = set()
        current_prods = {1}
        
        for x in order:
            new_prods = can_add(current_prods, x)
            if new_prods is not None:
                current_set.add(x)
                current_prods = current_prods | new_prods
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Local search: try to improve the best set by swapping elements
    if best_size < 18 and time.time() < deadline:
        best_set_set = set(best_set)
        # Compute subset products for best set
        best_prods = {1}
        for x in sorted(best_set_set):
            new_prods = set()
            for p in best_prods:
                new_prods.add(x * p)
            best_prods |= new_prods
        
        # Try removing one element and adding a different one
        for remove_x in list(best_set_set):
            if time.time() > deadline:
                break
            # Compute prods without remove_x
            temp_prods = {1}
            for x in sorted(best_set_set):
                if x == remove_x:
                    continue
                new_prods = set()
                for p in temp_prods:
                    new_prods.add(x * p)
                temp_prods |= new_prods
            
            for add_x in candidates:
                if add_x in best_set_set:
                    continue
                new_prods = can_add(temp_prods, add_x)
                if new_prods is not None:
                    # We found a swap that maintains validity
                    # This gives us the same size, but might allow further growth
                    new_set = (best_set_set - {remove_x}) | {add_x}
                    new_full_prods = temp_prods | new_prods
                    # Try to extend this new set
                    for ext_x in candidates:
                        if ext_x in new_set:
                            continue
                        ext_prods = can_add(new_full_prods, ext_x)
                        if ext_prods is not None:
                            new_set.add(ext_x)
                            new_full_prods |= ext_prods
                    if len(new_set) > best_size:
                        best_size = len(new_set)
                        best_set = list(new_set)
                        best_set_set = set(best_set)
                        best_prods = new_full_prods
                        break
    
    # Backtracking search for size 18
    if best_size < 18 and time.time() < deadline:
        # Try to find a set of size 18 via backtracking
        target_size = 18
        
        def backtrack(current_set, current_prods, remaining_candidates, depth):
            nonlocal best_size, best_set
            if time.time() > deadline:
                return False
            if len(current_set) == target_size:
                best_size = target_size
                best_set = list(current_set)
                return True
            if not remaining_candidates:
                return False
            if len(current_set) + len(remaining_candidates) < target_size:
                return False
            
            for i, x in enumerate(remaining_candidates):
                if time.time() > deadline:
                    return False
                new_prods = can_add(current_prods, x)
                if new_prods is not None:
                    current_set.add(x)
                    new_remaining = remaining_candidates[:i] + remaining_candidates[i+1:]
                    if backtrack(current_set, current_prods | new_prods, new_remaining, depth + 1):
                        return True
                    current_set.remove(x)
            return False
        
        # Use a heuristic: try primes first, then numbers with few prime factors
        def num_prime_factors(x):
            count = 0
            d = 2
            while d * d <= x:
                while x % d == 0:
                    count += 1
                    x //= d
                d += 1
            if x > 1:
                count += 1
            return count
        
        # Try several different orderings for backtracking
        orderings = [
            sorted(candidates, key=lambda x: (num_prime_factors(x), x)),
            sorted(candidates, key=lambda x: (x.bit_length(), x)),
            sorted(candidates, reverse=True),
            candidates[:],
        ]
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            # Only consider candidates not already in best_set for efficiency
            # Actually, let's just try the full backtracking with this ordering
            current_set = set()
            current_prods = {1}
            if backtrack(current_set, current_prods, ordering, 0):
                break
    
    # Return as a list of Python ints
    return sorted(best_set)

# EVOLVE-BLOCK-END
