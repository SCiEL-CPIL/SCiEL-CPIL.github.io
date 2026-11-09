import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    
    # Initialize S array
    # S[i] will store the part of i composed of primes < 24
    S = np.ones(max_val + 1, dtype=np.int64)
    
    # Fast NumPy Sieve
    for p in primes:
        power = p
        while power <= max_val:
            # Multiply S at indices power, 2*power, 3*power... by p
            # This effectively adds one factor of p to the exponent count
            S[power::power] *= p
            power *= p
            
    # Step 1: Check for distinct count 1
    # We need S[n+1] == S[n+2] == ... == S[n+24]
    # This implies S[n+1] == S[n+2], ..., S[n+23] == S[n+24]
    # Let D[i] = S[i+1] - S[i]
    # We need D[n+1] == 0, ..., D[n+23] == 0
    # That is 23 consecutive zeros in D starting at index n+1
    
    D = np.diff(S)
    # D has length max_val
    # We are interested in n in [1000, 1000000]
    # Indices in D corresponding to n+1 ... n+23
    # Start index in D: n+1
    # End index in D: n+23
    
    # Find runs of zeros in D
    is_zero = (D == 0)
    
    # To find runs of 23 zeros:
    # We can use convolution or just check windows.
    # But checking windows of 23 in a vectorized way:
    # Sum of is_zero[i:i+23] == 23
    
    # We can use np.convolve or just a sliding sum.
    # np.convolve(is_zero, np.ones(23), mode='valid')
    # This gives the sum of each window of size 23.
    # If sum == 23, all are zero.
    
    # However, we only care about n in [1000, 1000000].
    # Window start index in D is n+1.
    # So we check D[n+1 : n+24]
    
    # Let's compute the convolution for the whole array, then slice.
    # Convolution length: len(D) - 23 + 1
    # Index k in conv result corresponds to D[k : k+23]
    # We want D[n+1 : n+24], so k = n+1.
    
    conv_res = np.convolve(is_zero, np.ones(23, dtype=np.int8), mode='valid')
    
    # conv_res[i] corresponds to D[i : i+23]
    # We want i = n+1.
    # n ranges from 1000 to 1000000.
    # i ranges from 1001 to 1000001.
    
    target_indices = conv_res[1001:1000002]
    
    if np.any(target_indices == 23):
        # Found a window with distinct count 1
        # Get the first such index
        idx = np.argmax(target_indices == 23)
        n = 1000 + idx
        return n
        
    # Step 2: Search for distinct count 2
    # Since 1 is impossible, 2 is the best possible.
    # We can break as soon as we find 2.
    
    best_n = -1
    min_distinct = 3 # We know it's > 1, so start looking for 2
    
    # We can use a faster inner loop?
    # np.unique is still the best.
    
    S_local = S
    np_unique = np.unique
    
    for n in range(1000, max_n + 1):
        window = S_local[n+1:n+25]
        d = len(np_unique(window))
        if d == 2:
            return n
        elif d < min_distinct:
            min_distinct = d
            best_n = n
            
    return best_n
