import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Primes strictly below 24
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range for n is [1000, 1000000]
    # We need n+1 to n+24, so max index is 1000024
    min_n = 1000
    max_n = 1000000
    max_idx = max_n + 24
    
    # Precompute smooth parts
    # smooth[m] = product of prime powers p^k || m for p in P
    smooth = [0] * (max_idx + 1)
    
    for m in range(min_n + 1, max_idx + 1):
        x = m
        s = 1
        for p in P:
            while x % p == 0:
                x //= p
                s *= p
        smooth[m] = s
        
    # Sliding window
    best_n = min_n
    best_count = 25  # Max possible is 24, but 25 is safe init
    freq = {}
    
    # Initialize window for n = min_n
    # Window covers indices min_n+1 to min_n+24
    for i in range(min_n + 1, min_n + 25):
        val = smooth[i]
        freq[val] = freq.get(val, 0) + 1
        
    best_count = len(freq)
    best_n = min_n
    
    # Slide window
    for n in range(min_n + 1, max_n + 1):
        if time.time() > deadline:
            break
            
        # Remove n
        old_val = smooth[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            
        # Add n+24
        new_val = smooth[n + 24]
        freq[new_val] = freq.get(new_val, 0) + 1
        
        current_count = len(freq)
        if current_count < best_count:
            best_count = current_count
            best_n = n
            
            # If we found 2, it's likely the minimum possible (since 1 is impossible)
            # We could break early if best_count == 2, but let's check if 2 is possible.
            # Actually, if we find 2, we can break to save time, or continue to ensure no 1? 
            # 1 is impossible. So 2 is the global minimum.
            if best_count == 2:
                # We can break here as we cannot do better than 2
                # But let's be safe and just break.
                break

    return best_n
