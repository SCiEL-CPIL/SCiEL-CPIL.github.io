import numpy as np

def construct():
    MAX_N = 1000000
    OFFSET = 24
    MAX_VAL = MAX_N + OFFSET
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth parts for all numbers up to MAX_VAL
    # smooths[i] will hold the product of p^k for p < 24 dividing i
    smooths = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in primes:
        power = p
        while power <= MAX_VAL:
            # For each multiple of power, multiply smooths by p
            # This effectively adds one factor of p for each time p^k divides the number
            # Wait, if i is divisible by p^2, it is a multiple of p and p^2.
            # Loop 1 (power=p): smooths[i] *= p
            # Loop 2 (power=p^2): smooths[i] *= p
            # Total multiplication: p * p = p^2. Correct.
            indices = np.arange(power, MAX_VAL + 1, power)
            smooths[indices] *= p
            power *= p
            
    best_n = 1000
    best_count = 25 # Max distinct is 24, but 25 is safe upper bound
    
    # Iterate through valid n
    # n ranges from 1000 to MAX_N - OFFSET
    # Window is n+1 to n+24 (inclusive)
    # Indices in smooths: n+1 to n+24
    
    for n in range(1000, MAX_N - OFFSET + 1):
        # Extract the window
        # Using set on numpy slice creates a set of numpy scalars, which might be slower than python ints
        # But 10^6 iterations * 24 elements is ~24M operations.
        # Let's try to optimize the inner loop.
        
        # A python loop with set() on a list slice is often faster than numpy set operations for small arrays
        # because numpy overhead per operation is high.
        # However, accessing numpy array elements is fast.
        
        window = smooths[n+1 : n+25]
        # Convert to python list/set to avoid numpy scalar overhead in set comparison?
        # Actually, len(set(window)) on a numpy array works.
        count = len(set(window))
        
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
