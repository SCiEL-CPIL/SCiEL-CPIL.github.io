from collections import Counter

def construct():
    MAX_N = 1000000
    MAX_M = MAX_N + 24
    primes_below_24 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    smooth = [1] * (MAX_M + 1)
    for p in primes_below_24:
        for multiple in range(p, MAX_M + 1, p):
            count = 0
            temp = multiple
            while temp % p == 0:
                count += 1
                temp //= p
            if count > 0:
                smooth[multiple] *= p ** count
    
    best_n = 1000
    best_count = 25
    
    # Sliding window with frequency counter
    window_freq = Counter()
    for i in range(1, 25):
        window_freq[smooth[1000 + i]] += 1
    best_count = len(window_freq)
    best_n = 1000
    
    for n in range(1001, MAX_N + 1):
        # Remove smooth[n] (which was n+1 in previous window... wait, let me re-index)
        # Window for n-1 was [n, n+23] (smooth[n] to smooth[n+23])
        # Window for n is [n+1, n+24] (smooth[n+1] to smooth[n+24])
        # So remove smooth[n], add smooth[n+24]
        val_remove = smooth[n]
        val_add = smooth[n + 24]
        
        window_freq[val_remove] -= 1
        if window_freq[val_remove] == 0:
            del window_freq[val_remove]
        
        window_freq[val_add] += 1
        
        count = len(window_freq)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break
    
    return best_n
