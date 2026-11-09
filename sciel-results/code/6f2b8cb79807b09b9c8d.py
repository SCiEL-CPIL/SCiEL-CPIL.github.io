def construct():
    import time
    start_time = time.time()
    limit = 1000000
    N = limit + 1
    
    # Compute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_val = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 1000000]
    # We can break early if we find a very small value?
    # The minimum possible is 2 (if n=2, n+1=3? No n>=10000).
    # Smallest prime is 2.
    # If we find value 2, we can stop?
    # n and n+1 coprime. One even, one odd.
    # If n is even, LPF(n) >= 2. If n+1 is odd, LPF(n+1) >= 3 (unless n+1=1, impossible).
    # So min possible max is at least 3?
    # If n=2, n+1=3, max(2,3)=3.
    # For n>=10000, can we get max=3?
    # n and n+1 must be 3-smooth.
    # 3-smooth numbers: 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243, 256, 288, 324, 384, 432, 486, 512, 576, 648, 729, 768, 864, 972, 1024, 1152, 1296, 1458, 1536, 1728, 1944, 2048, 2304, 2592, 2916, 3072, 3456, 3888, 4096, 4608, 5184, 5832, 6144, 6912, 7776, 8192, 9216, 10368, 11664, 12288, 13824, 15552, 16384, 18432, 20736, 23328, 26214, 29491, 32768, 36864, 41472, 46656, 52428, 58982, 65536, 73728, 82944, 93312, 10485, 11796, 13107, 14745, 16588, 18604, 20971, 23592, 26214, 29491, 32768...
    # Consecutive 3-smooth numbers?
    # 1,2; 2,3; 3,4; 4,6? No; 8,9; 15? No; 24, 27? No.
    # It is known that there are no consecutive 3-smooth numbers larger than 4 and 6?
    # Actually, 8 and 9 are consecutive.
    # Are there larger ones?
    # Catalan's conjecture (Mihăilescu's theorem) says only 8 and 9 are consecutive perfect powers.
    # But 3-smooth includes non-powers like 12, 18.
    # Gap between 3-smooth numbers increases.
    # It is unlikely to have consecutive 3-smooth numbers $> 9$.
    # So min value is likely > 3.
    # We will just run the full search. It's fast.
    
    for n in range(10000, limit + 1):
        # Check time
        if time.time() - start_time > 50:
            break
        v1 = lpf[n]
        v2 = lpf[n+1]
        v = v1 if v1 > v2 else v2
        if v < best_val:
            best_val = v
            best_n = n
            # If we hit a very low value, we might stop?
            # But we don't know the lower bound.
            # Just finish the loop.
            
    return best_n
