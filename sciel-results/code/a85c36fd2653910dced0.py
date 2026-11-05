import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115  # 5 seconds buffer before 120s limit
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Search for a window of length 200 with distinct totients
    # We search near 10^12 because collisions are less frequent for large numbers.
    max_val = 10**12
    window_len = 200
    
    # We will try windows starting from max_val - window_len + 1 downwards.
    # We check if phi(s), ..., phi(s+199) are distinct.
    
    best_start = 1
    best_len = 1
    
    # We can check multiple windows in a batch or one by one.
    # One by one is simple and fast enough if the first few work.
    
    # Optimization: Pre-allocate arrays for the window
    phi_vals = [0] * window_len
    rem_vals = [0] * window_len
    
    # Start searching from the end
    # s + window_len - 1 <= 10^12 => s <= 10^12 - 199
    current_s = max_val - window_len + 1
    
    # We might not need to check every single s. 
    # But checking is fast.
    # Let's iterate s downwards.
    
    found = False
    checked = 0
    
    while current_s >= 1:
        if time.time() > deadline:
            break
            
        # Initialize for this window
        for i in range(window_len):
            val = current_s + i
            phi_vals[i] = val
            rem_vals[i] = val
            
        # Apply sieve for each prime
        # For each prime p, we need to find the first multiple in [current_s, current_s + window_len - 1]
        # first_multiple = ceil(current_s / p) * p
        
        for p in primes:
            # If p is larger than the max value in the window, we can stop?
            # No, p is a factor. If p > window_len, each number has at most one factor p.
            # If p*p > max_val, we handle the remainder later.
            # Actually, we just iterate all primes up to 10^6.
            
            # Find first multiple of p >= current_s
            # start_idx = (p - current_s % p) % p
            # This gives the offset from current_s to the first multiple.
            offset = (p - current_s % p) % p
            
            # Iterate through multiples in the window
            # Indices: offset, offset + p, offset + 2p, ... < window_len
            for j in range(offset, window_len, p):
                # rem_vals[j] is the current remaining part of (current_s + j)
                if rem_vals[j] % p == 0:
                    phi_vals[j] = phi_vals[j] // p * (p - 1)
                    while rem_vals[j] % p == 0:
                        rem_vals[j] //= p
        
        # Handle remaining prime factors > 10^6
        # If rem_vals[j] > 1, it is a prime factor (since we divided by all primes <= 10^6)
        for j in range(window_len):
            if rem_vals[j] > 1:
                phi_vals[j] = phi_vals[j] // rem_vals[j] * (rem_vals[j] - 1)
        
        # Check for distinctness
        # Use a set
        if len(set(phi_vals)) == window_len:
            return {'start': current_s, 'length': window_len}
            
        # Move to next window
        current_s -= 1
        checked += 1
        
        # If we don't find it in the first 1000 windows, something might be wrong,
        # but theoretically it should be very frequent.
        # We continue until deadline.
        
    # Fallback: return best found or a valid small one
    # If we didn't find 200, we might have found less? 
    # The logic above only checks for exactly 200.
    # If we fail, we should probably return a known valid small one or try to find max k < 200.
    # However, k=200 is expected to be found.
    
    # To be safe, let's return a hardcoded valid result if search fails, 
    # or just the best effort. 
    # Actually, the problem says "Maximize k". 
    # If we didn't find 200, we should probably search for smaller k? 
    # But 200 is the cap. 
    # Given the evidence, 200 is found.
    
    return {'start': best_start, 'length': best_len}
# EVOLVE-BLOCK-END
