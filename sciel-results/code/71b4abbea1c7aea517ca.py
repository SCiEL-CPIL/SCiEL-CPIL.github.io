import numpy as np

def construct():
    # Upper bound for the search space
    upper_limit = 1000000
    lower_limit = 10000
    
    # Array to store the largest prime factor for each number
    # Size is upper_limit + 1 to include the upper bound itself
    max_prime_factor = np.zeros(upper_limit + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes variant to find largest prime factors
    # Iterate through potential prime factors
    for candidate in range(2, upper_limit + 1):
        # If max_prime_factor[candidate] is 0, candidate is prime
        if max_prime_factor[candidate] == 0:
            # Update all multiples of this prime
            # Since we iterate in increasing order, the last update
            # for any number will be by its largest prime factor
            max_prime_factor[candidate::candidate] = candidate
            
    # We need to find n in [lower_limit, upper_limit]
    # such that max(lpf(n), lpf(n+1)) is minimized.
    # Note: n+1 can be up to upper_limit + 1.
    # So we need lpf computed up to upper_limit + 1.
    # Our array size is upper_limit + 1, so indices 0..upper_limit.
    # We need index upper_limit + 1?
    # If n = upper_limit, we need lpf(upper_limit + 1).
    # The current array size is upper_limit + 1, so max index is upper_limit.
    # We are missing lpf(upper_limit + 1).
    
    # Let's fix the array size.
    # Max n is 1000000. Max n+1 is 1000001.
    # So we need array size 1000002.
    
    max_val = upper_limit + 1
    max_prime_factor = np.zeros(max_val + 1, dtype=np.int64)
    
    for candidate in range(2, max_val + 1):
        if max_prime_factor[candidate] == 0:
            max_prime_factor[candidate::candidate] = candidate
            
    # Extract the relevant slices
    # lpf_n: lpf[lower_limit] to lpf[upper_limit]
    # lpf_np1: lpf[lower_limit+1] to lpf[upper_limit+1]
    
    lpf_n = max_prime_factor[lower_limit : upper_limit + 1]
    lpf_np1 = max_prime_factor[lower_limit + 1 : max_val + 1]
    
    # Compute the maximum of the two for each n
    combined_max = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_index = np.argmin(combined_max)
    
    # The corresponding n
    best_n = lower_limit + min_index
    
    return int(best_n)
