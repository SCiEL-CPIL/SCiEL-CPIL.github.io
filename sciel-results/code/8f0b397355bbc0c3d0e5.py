import time

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24
    
    # Precompute small prime parts
    small_parts = [1] * (max_val + 1)
    
    for p in primes:
        p_power = p
        while p_power <= max_val:
            next_power = p_power * p
            for m in range(p_power, max_val + 1, p_power):
                if m % next_power != 0:
                    small_parts[m] *= p
            p_power = next_power

    best_n = 1000
    best_score = 25
    
    # Use a list for counts for O(1) access speed
    # Max value of small_parts is max_val
    counts = [0] * (max_val + 1)
    
    distinct_count = 0
    # Initialize window for n=1000: [1001, 1024]
    for i in range(1, 25):
        val = small_parts[1000 + i]
        if counts[val] == 0:
            distinct_count += 1
        counts[val] += 1
            
    if distinct_count < best_score:
        best_score = distinct_count
        best_n = 1000
        
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    # Local variables for speed
    sp = small_parts
    cnt = counts
    dc = distinct_count
    bs = best_score
    bn = best_n
    
    # Iterate n from 1001 to 1000000
    for n in range(1001, 1000001):
        # Check time periodically to avoid overhead
        if (n & 1023) == 0:
            if time.time() > deadline:
                break
                
        # Remove element at n (which was n-1 + 1)
        val_out = sp[n]
        cnt[val_out] -= 1
        if cnt[val_out] == 0:
            dc -= 1
            
        # Add element at n+24
        val_in = sp[n + 24]
        if cnt[val_in] == 0:
            dc += 1
        cnt[val_in] += 1
        
        if dc < bs:
            bs = dc
            bn = n
            if bs == 1:
                break
                
    return bn
