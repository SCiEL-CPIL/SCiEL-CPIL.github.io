import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second safety margin (90s limit)
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range of numbers we need: n+1 to n+24 where n in [1000, 1000000]
    # So numbers from 1001 to 1000024
    min_num = 1001
    max_num = 1000024
    
    # Precompute extracted factors using sieve-like approach
    size = max_num - min_num + 1
    f = [1] * size
    
    # For each prime, divide out all factors
    for p in primes:
        for i in range(min_num, max_num + 1):
            if i % p == 0:
                m = i
                while m % p == 0:
                    f[i - min_num] *= p
                    m //= p
    
    # Sliding window with optimized distinct count
    best_n = 1000
    best_count = float('inf')
    
    # Initialize window for n=1000: f[0:24]
    freq = {}
    for val in f[0:24]:
        freq[val] = freq.get(val, 0) + 1
    distinct = len(freq)
    
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide window
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        remove_val = f[n - min_num]
        add_val = f[n + 24 - min_num]
        
        # Remove
        cnt = freq[remove_val]
        if cnt == 1:
            del freq[remove_val]
            distinct -= 1
        else:
            freq[remove_val] = cnt - 1
        
        # Add
        if add_val in freq:
            freq[add_val] += 1
        else:
            freq[add_val] = 1
            distinct += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count <= 2:
                break
    
    return best_n
