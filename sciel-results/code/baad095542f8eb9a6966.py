import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}
START_TIME = time.time()
DEADLINE = 60.0

def construct():
    global START_TIME, DEADLINE
    
    candidates = list(range(2, P['n'] + 1))
    
    # Pre-compute factorization vectors
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    vectors = {}
    for num in candidates:
        f = factorize(num)
        vectors[num] = tuple(f.get(p, 0) for p in primes)

    # Check if a set of numbers is 1-independent
    # i.e., all subset products are distinct
    def is_valid_subset(subset):
        # Generate all subset products
        # If len(subset) > 20, this is too slow.
        # But we expect size ~15-18. 2^18 = 262144.
        # This is feasible.
        products = {1}
        for x in subset:
            new_prods = {p * x for p in products}
            if new_prods & products:
                return False
            products |= new_prods
        return True

    # Greedy initialization
    # Sort by "simplicity" or just try to build a large set
    # Heuristic: Start with primes, then add composites that don't create collisions
    
    # Better: Branch and Bound
    # We want to maximize the size of the subset.
    
    best_subset = []
    
    # Sort candidates to prioritize those likely to be independent
    # Primes are "pure", composites are "mixed"
    # Actually, mixing primes can allow more composites?
    # No, primes are the basis.
    # Let's just sort by value.
    
    def backtrack(start, current_subset, current_products):
        nonlocal best_subset
        
        if time.time() > DEADLINE - 5.0:
            return
            
        if len(current_subset) > len(best_subset):
            best_subset = current_subset[:]
            
        # Pruning: if we can't beat the best with remaining items
        # Max possible size = len(current_subset) + (len(candidates) - start)
        if len(current_subset) + (len(candidates) - start) <= len(best_subset):
            return
            
        for i in range(start, len(candidates)):
            if time.time() > DEADLINE - 5.0:
                return
                
            num = candidates[i]
            # Check if num can be added
            # New products = {p * num for p in current_products}
            new_prods = {p * num for p in current_products}
            
            if not (new_prods & current_products):
                # Valid
                backtrack(i + 1, current_subset + [num], current_products | new_prods)
                
    # Start backtracking
    # Initial state: empty subset, products={1}
    # To speed up, we can sort candidates.
    # Let's try sorting by number of prime factors (ascending)
    def complexity(n):
        return sum(vectors[n])
        
    sorted_candidates = sorted(candidates, key=complexity)
    
    # Replace candidates in backtrack with sorted_candidates
    # We need to update the global reference or pass it
    
    # Let's use the sorted list
    backtrack(0, [], {1})
    
    # Ensure the result is a list of ints
    return best_subset

# EVOLVE-BLOCK-END
