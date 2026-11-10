import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = 85.0  # Safety margin for 90s limit
    
    S = 123457
    
    # Generate primes up to S
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(S)
    
    best_result = [1, S]
    best_score = 109 # P(123456) = 109
    
    # We iterate through primes as the bound B
    # For each B, we check if there is a decomposition into B-smooth numbers
    # If so, the max prime factor is at most B.
    # Since we iterate B in increasing order, the first valid B is the minimum possible max prime factor.
    
    # To generate B-smooth numbers <= S, we use a recursive approach or BFS
    # Let's use a set to store generated smooth numbers
    
    # We can optimize by checking if time is running out
    for idx, B in enumerate(primes):
        if time.time() - start_time > limit_time:
            break
            
        # Generate all B-smooth numbers <= S
        # Primes to use are primes[:idx+1]
        current_primes = primes[:idx+1]
        
        # Efficient generation using a list-based approach with early termination
        # We maintain a sorted list of smooth numbers and generate new ones
        # using a multi-pointer technique or simple DFS with pruning
        
        # Use a set for membership testing and a list for iteration
        smooth_set = {1}
        smooth_list = [1]
        
        # For each prime in current_primes, extend the set
        # We do this iteratively: for each prime p, multiply existing elements by p^k
        # until we exceed S
        
        for p in current_primes:
            # Get current size to avoid processing newly added elements in this round
            current_size = len(smooth_list)
            for i in range(current_size):
                x = smooth_list[i]
                nx = x * p
                while nx <= S:
                    if nx not in smooth_set:
                        smooth_set.add(nx)
                        smooth_list.append(nx)
                    nx *= p
        
        # Check for a pair
        # We can iterate over the smaller of the set or just iterate a from 1 to S//2
        # Iterating over the set is faster if the set is small
        for a in smooth_set:
            b = S - a
            if b in smooth_set:
                # Found a valid pair with max prime factor <= B
                # Since B is the current prime bound, P(a) <= B and P(b) <= B
                # Is it possible that P(a) < B or P(b) < B? Yes.
                # But the max is at most B.
                # Since we iterate B in increasing order, this is the minimal possible max value.
                # We should return this pair.
                return [a, b]
                
        # If we didn't find a pair for this B, continue to next prime
        # Note: It's possible that a pair exists with max prime factor exactly B, 
        # or less than B (if we missed it in previous steps? No, previous steps covered smaller primes).
        # If a pair has max prime factor K, it will be found when B=K.
        
    # If no pair found within time, return the best known or seed
    return best_result
# EVOLVE-BLOCK-END
