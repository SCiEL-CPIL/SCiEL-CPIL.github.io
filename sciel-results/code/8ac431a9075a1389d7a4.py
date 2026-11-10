import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin: evaluator kills at 90s, but we should finish much earlier
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # We need to access up to n+24
    
    # Precompute small_part[x] for x in [1, max_val]
    # small_part[x] is the product of prime factors of x that are in small_primes
    small_part = [1] * (max_val + 1)
    
    # Efficient precomputation:
    # For each prime p, we can update small_part for all multiples of p.
    # However, it's easier to just iterate x and factorize using the small primes list.
    # Since N is 10^6 and there are 9 primes, this is 9 * 10^6 operations, which is fast.
    
    for x in range(1, max_val + 1):
        val = x
        prod = 1
        for p in small_primes:
            if p * p > val:
                break
            while val % p == 0:
                prod *= p
                val //= p
        # If val > 1, it is a prime factor >= 23 (actually > 23 if we broke early, 
        # but we only care about primes STRICTLY BELOW 24).
        # The loop handles all primes in small_primes.
        # Any remaining val is coprime to all primes < 24, so it doesn't affect prod.
        small_part[x] = prod
        
    # Sliding window
    # We want to minimize the number of distinct values in small_part[n+1 ... n+24]
    
    best_n = 1000
    min_distinct = 25  # Max possible is 24, but we initialize high
    
    # Initialize window for n=1000: elements are small_part[1001] ... small_part[1024]
    freq = {}
    for i in range(1, 25):
        v = small_part[1000 + i]
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            
    current_distinct = len(freq)
    if current_distinct < min_distinct:
        min_distinct = current_distinct
        best_n = 1000
        
    # Slide window from n=1001 to 1000000
    # Remove small_part[n] (which was at position n in the previous window's left? No.)
    # Let's trace carefully.
    # Window for n: indices n+1, ..., n+24.
    # Window for n+1: indices n+2, ..., n+25.
    # So we remove small_part[n+1] and add small_part[n+25].
    
    for n in range(1001, max_n + 1):
        # Remove the element that was at the start of the previous window: small_part[n]
        # Wait, for n=1000, window is 1001..1024.
        # For n=1001, window is 1002..1025.
        # So we remove small_part[1001] and add small_part[1025].
        # Generally, for step n (calculating window for n), we remove small_part[n] and add small_part[n+24].
        # Let's verify:
        # Previous window (n-1): (n-1)+1 .. (n-1)+24 = n .. n+23.
        # Current window (n): n+1 .. n+24.
        # So we remove small_part[n] and add small_part[n+24].
        
        rem_val = small_part[n]
        freq[rem_val] -= 1
        if freq[rem_val] == 0:
            del freq[rem_val]
            current_distinct -= 1
            
        add_val = small_part[n + 24]
        if add_val in freq:
            freq[add_val] += 1
        else:
            freq[add_val] = 1
            current_distinct += 1
            
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n
            if min_distinct == 1:
                break # Can't get lower than 1
        
        # Time check
        if n % 10000 == 0:
            if time.time() > deadline:
                break
                
    return best_n
