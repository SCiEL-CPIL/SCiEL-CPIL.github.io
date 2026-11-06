import time
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    
    # Define candidate prime sets
    # We want sets with sum(1/p) <= 1.
    # Strategy: Use small primes for high density, fill budget with large primes for flexibility.
    
    candidate_sets = []
    
    # Set 1: {2, 3, 7, 41}
    # Sum: 1/2 + 1/3 + 1/7 + 1/41 = 0.9995
    candidate_sets.append([2, 3, 7, 41])
    
    # Set 2: {2, 3, 7, 43}
    # Sum: 1/2 + 1/3 + 1/7 + 1/43 = 0.9990
    candidate_sets.append([2, 3, 7, 43])
    
    # Set 3: {2, 3, 7, 47}
    # Sum: 1/2 + 1/3 + 1/7 + 1/47 = 0.9984
    candidate_sets.append([2, 3, 7, 47])
    
    # Set 4: {2, 3, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    # This sum is way too high.
    
    # Let's try a set with more primes but slightly larger.
    # {2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    # Sum is ~2.1.
    
    # How about {2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}?
    # Sum(2,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97)
    # Sum(2..97) = 2.19
    # Sum(3) = 0.333
    # Sum(4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28,30,32,33,34,35,36,38,39,40,42,44,45,46,48,49,50,51,52,54,55,56,57,58,60,62,63,64,65,66,68,69,70,72,74,75,76,77,78,80,81,82,84,85,86,87,88,90,91,92,93,94,95,96,98,99,100)
    # This is getting complicated.
    
    # Let's just stick to the small prime sets.
    # If {2, 3, 7, 41} fails, try {2, 3, 7, 43}, etc.
    # Also try {2, 3, 5, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    # Sum(2,3,5) = 1.033. Too big.
    
    # What about {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}?
    # Sum(3..97) = 2.19 - 0.5 = 1.69. Too big.
    
    # It seems we are limited to very few primes if we want small ones.
    # Or many large ones.
    # But small ones are better for coverage.
    
    # Let's add a few more candidates.
    # {2, 3, 7, 41, 43} -> Sum 0.9995 + 0.023 = 1.02. Too big.
    # {2, 3, 7, 47, 53} -> Sum 0.9984 + 0.0188 = 1.017. Too big.
    # {2, 3, 7, 53} -> Sum 0.9984 - 0.0243 + 0.0188 = 0.9929.
    candidate_sets.append([2, 3, 7, 53])
    
    # {2, 3, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    # Sum(2,3,11..97) = 2.19 - 1/5 - 1/7 = 2.19 - 0.2 - 0.142 = 1.848. Too big.
    
    # It appears {2, 3, 7, p} is the main family.
    
    best_solution = None
    best_L = 0
    
    for p_set in candidate_sets:
        # Check budget
        budget_sum = sum(1.0 / p for p in p_set)
        if budget_sum > 1.0:
            continue
            
        # Search for residues
        # If the product of primes is small, brute force.
        # Product of {2, 3, 7, 41} is 1722.
        # Product of {2, 3, 7, 53} is 2214.
        # Brute force is feasible.
        
        n_primes = len(p_set)
        ranges = [range(p) for p in p_set]
        
        # Use itertools.product for brute force if size is small
        # 1722 * 2214 * ... is small enough.
        
        import itertools
        
        for residues in itertools.product(*ranges):
            # Check if this combination covers 1..1000
            covered = [False] * 1001
            for p, r in zip(p_set, residues):
                for n in range(1, 1001):
                    if n % p == r:
                        covered[n] = True
            
            # Find max L
            L = 0
            for n in range(1, 1001):
                if covered[n]:
                    L = n
                else:
                    break
            
            if L > best_L:
                best_L = L
                best_solution = [[p, r] for p, r in zip(p_set, residues)]
                
            # Early termination if we reach 999
            if best_L >= 999:
                return best_solution
                
            # Time check
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    if best_solution is None:
        # Fallback to seed-like solution if search fails (should not happen)
        primes_seed = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_solution
