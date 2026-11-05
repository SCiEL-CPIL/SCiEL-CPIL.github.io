from collections import defaultdict

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    min_n = 1000
    window_size = 24
    
    start_val = min_n + 1
    end_val = max_n + window_size
    
    # Precompute smooth parts for all numbers up to end_val
    # smooths[x] is the factor of x composed of primes < 24
    smooths = [0] * (end_val + 1)
    
    for x in range(1, end_val + 1):
        s = x
        for p in primes:
            while s % p == 0:
                s //= p
        # s is now the rough part (coprime to primes < 24)
        smooths[x] = x // s

    # Sliding window to find min distinct smooth parts in [n+1, n+24]
    freq = defaultdict(int)
    
    # Initialize first window for n = min_n
    # Range: [min_n+1, min_n+24]
    for i in range(min_n + 1, min_n + 25):
        freq[smooths[i]] += 1
        
    min_distinct = len(freq)
    best_n = min_n
    
    # Slide window for n from min_n+1 to max_n
    for n in range(min_n + 1, max_n + 1):
        # Remove smooths[n] (leftmost of previous window)
        out_val = smooths[n]
        freq[out_val] -= 1
        if freq[out_val] == 0:
            del freq[out_val]
            
        # Add smooths[n+24] (new rightmost)
        in_val = smooths[n + 24]
        freq[in_val] += 1
        
        current_distinct = len(freq)
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n
            
    return best_n
