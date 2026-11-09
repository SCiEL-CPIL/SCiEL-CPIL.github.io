import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range of numbers we need: n+1 to n+24 where n in [1000, 1000000]
    # So numbers from 1001 to 1000024
    min_num = 1001
    max_num = 1000024
    
    # Precompute extracted factors
    # f[i] = extracted factor of i
    # We'll use an array indexed by i - min_num
    size = max_num - min_num + 1
    f = [0] * size
    
    for i in range(min_num, max_num + 1):
        m = i
        result = 1
        for p in primes:
            while m % p == 0:
                result *= p
                m //= p
        f[i - min_num] = result
    
    # Sliding window
    from collections import Counter
    
    best_n = 1000
    best_count = float('inf')
    
    # Window covers f[n+1] to f[n+24], indices (n+1-min_num) to (n+24-min_num)
    # For n=1000: indices 0 to 23
    # For n=1000000: indices 998999 to 999022
    
    # Initialize window for n=1000
    window = f[0:24]
    freq = Counter(window)
    distinct = len(freq)
    
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide
    for n in range(1001, 1000001):
        # Remove f[n+1 - 1 - min_num] = f[n - min_num] ... wait let me re-index
        
        # Window for n: f[n+1] to f[n+24]
        # Window for n-1: f[n] to f[n+23]
        # So when going from n-1 to n: remove f[n], add f[n+24]
        
        # f[n] is at index n - min_num
        # f[n+24] is at index n+24 - min_num
        
        if time.time() > deadline:
            break
        
        remove_val = f[n - min_num]  # f[n]
        add_val = f[n + 24 - min_num]  # f[n+24]
        
        freq[remove_val] -= 1
        if freq[remove_val] == 0:
            del freq[remove_val]
            distinct -= 1
        
        if add_val in freq:
            freq[add_val] += 1
        else:
            freq[add_val] = 1
            distinct += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break
    
    return best_n
