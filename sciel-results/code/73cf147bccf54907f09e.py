import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # We want to find n such that max(LPF(n), LPF(n+1)) is minimized.
    # We iterate y from 2 upwards.
    # For each y, we check if there is an n in [N_MIN, N_MAX]
    # such that n is y-smooth and n+1 is y-smooth.
    
    # Pre-allocate a bytearray for smoothness check to avoid list overhead?
    # A list of bools is fine.
    size = N_MAX + 2  # To check n+1 up to 1000001
    
    for y in range(2, 100):
        if time.time() > deadline:
            # Fallback: return a known good or just N_MIN
            # We should have found something by now.
            return N_MIN
        
        # Get primes <= y
        primes_y = [p for p in primes if p <= y]
        
        # Check if any n is y-smooth and n+1 is y-smooth
        # We can do this by iterating n and checking both.
        # To be efficient, we can precompute smoothness for all numbers in range.
        # But doing it inside the y-loop means recomputing for each y.
        # Is it better to compute LPF once?
        # Let's stick to the y-loop as it's likely to terminate very early.
        
        # Optimization: 
        # If y is small, few numbers are smooth.
        # We can generate the smooth numbers directly?
        # No, trial division is simpler and fast enough for small y.
        
        # Let's check n from N_MIN to N_MAX.
        # We can break early if we find one.
        
        # To speed up, we can check n and n+1 together.
        # But we need to know if n is smooth.
        
        # Let's use a helper to check if a number is y-smooth given primes_y.
        # But calling a function 10^6 times is slow.
        # Inline the logic.
        
        found = False
        # We can optimize by checking only if n is smooth, then check n+1.
        # Or check both in parallel?
        
        for n in range(N_MIN, N_MAX + 1):
            if time.time() > deadline:
                return N_MIN
            
            # Check if n is y-smooth
            temp_n = n
            is_smooth_n = True
            for p in primes_y:
                while temp_n % p == 0:
                    temp_n //= p
                if temp_n == 1:
                    break
            if temp_n != 1:
                continue
            
            # n is y-smooth. Now check n+1
            temp_np1 = n + 1
            is_smooth_np1 = True
            for p in primes_y:
                while temp_np1 % p == 0:
                    temp_np1 //= p
                if temp_np1 == 1:
                    break
            if temp_np1 == 1:
                # Both are y-smooth
                return n
                
    # If no small y found, fallback to a better heuristic or just return N_MIN
    # Given the density, this should not happen for y < 100.
    return N_MIN
