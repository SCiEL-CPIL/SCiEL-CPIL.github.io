import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    # Safety margin: 110s
    deadline = start_time + 110.0
    
    # Strategy: Use FFT to count sums of powerful numbers.
    # This is O(L log L) vs O(M^2) for block method.
    # Allows larger L within time limit.
    
    # Determine L based on memory and time.
    # FFT of size 2*L.
    # For L=2*10^7, size 4*10^7. 
    # np.fft.fft on 4*10^7 floats takes ~1-2 seconds? 
    # Generation of powerful numbers is O(L^(1/2) + L^(1/3)).
    
    # Let's try L = 2 * 10^7 first. If fast, increase.
    # But we have a fixed function. We can check time and if fast, do a second pass with larger L?
    # No, we must return a result.
    # Let's pick a conservative L that fits in memory and time, e.g., 5 * 10^6 or 10^7.
    # The cell finding said L=2e8 worked for a different strategy (maybe not FFT).
    # With FFT, L=10^7 is very fast. L=5*10^7 might be okay.
    
    # Let's start with L = 10^7.
    L = 10**7
    
    # Generate powerful numbers up to L
    # Powerful: x^2 * y^3, y square-free.
    
    # Primes up to L^(1/3) for square-free check? 
    # Actually, we just need to generate them.
    # Iterate y from 1 to L^(1/3).
    # Check if y is square-free.
    # For each square-free y, iterate x from 1 to sqrt(L/y^3).
    # val = x^2 * y^3.
    
    max_y = int(L ** (1.0/3.0)) + 2
    
    # Sieve for square-free? 
    # Or just check divisibility by p^2 for p up to sqrt(y).
    # Since y <= 2000 (for L=10^7, 10^7^(1/3) ~ 215), sqrt(y) is small.
    
    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % (p * p) == 0:
                return False
        return True
        
    powerful_bool = np.zeros(L + 1, dtype=np.int8)
    
    for y in range(1, max_y + 1):
        if not is_square_free(y):
            continue
        y3 = y ** 3
        if y3 > L:
            break
        max_x = int(math.isqrt(L // y3))
        for x in range(1, max_x + 1):
            val = x * x * y3
            if val <= L:
                powerful_bool[val] = 1
                
    # Convert to float array for FFT
    freq = powerful_bool.astype(np.float64)
    
    # FFT size
    # We need convolution up to 2*L.
    # Next power of 2 for FFT efficiency?
    # N_fft = 2 * (L + 1)
    # Or next power of 2 >= 2*(L+1)
    
    size = 2 * (L + 1)
    # Round up to next power of 2 for speed?
    # np.fft handles non-powers of 2 but they are slower.
    # 2*10^7 is not a power of 2.
    # Next power of 2 for 2*10^7 is 2^25 = 33,554,432.
    # This is manageable.
    
    n_fft = 1
    while n_fft < size:
        n_fft *= 2
        
    # Perform FFT
    fft_freq = np.fft.fft(freq, n=n_fft)
    conv = np.fft.ifft(fft_freq * fft_freq).real
    
    # conv[n] is the number of pairs (a,b) such that a+b=n.
    # We only care about n <= 2*L.
    # conv is an array of length n_fft.
    
    # Find the best N.
    # Score for N is (conv[N] + Y) // 2, where Y is 1 if N/2 is powerful.
    
    # We need to check N from 2 to 2*L.
    # conv[0] and conv[1] are 0 (min sum is 1+1=2? No, 1 is powerful. 1+1=2).
    
    max_score = 0
    best_N = 0
    
    # Iterate through possible N.
    # To speed up, we can find the max of conv first, then check neighbors?
    # The max of conv is likely close to the max score.
    # But we need exact score.
    
    # Let's just iterate. 2*10^7 iterations in Python is slow (~1s).
    # We can use NumPy to compute scores for all N at once.
    
    # Y vector: Y[N] = 1 if N is even and N/2 is powerful.
    # N/2 is powerful iff powerful_bool[N//2] == 1.
    # We need to handle odd N (Y=0).
    
    # Create Y array
    # N ranges from 0 to 2*L.
    # N is even if N % 2 == 0.
    # Let's create an array of indices 0..2*L.
    # This might be memory heavy if we create full arrays.
    
    # Alternative:
    # max_conv = np.max(conv[:2*L+1])
    # The best N is likely near the argmax.
    # But we need to be sure.
    
    # Let's compute the score for all N using NumPy vectorization.
    # N = np.arange(2, 2*L + 1)
    # This creates an array of size 2*10^7. 160MB. Okay.
    
    N_arr = np.arange(2, 2 * L + 1, dtype=np.int64)
    conv_part = conv[2:2 * L + 1]
    
    # Calculate Y
    # N_arr % 2 == 0
    # For even N, check powerful_bool[N//2]
    # For odd N, Y=0
    
    # We can do:
    # Y = np.zeros_like(N_arr, dtype=np.float64)
    # even_mask = (N_arr % 2 == 0)
    # Y[even_mask] = powerful_bool[N_arr[even_mask] // 2]
    
    # This is efficient.
    
    Y = np.zeros(len(N_arr), dtype=np.float64)
    even_mask = (N_arr % 2 == 0)
    # N_arr[even_mask] // 2 gives indices for powerful_bool
    # powerful_bool is int8, will be cast to float.
    Y[even_mask] = powerful_bool[N_arr[even_mask] // 2]
    
    scores = (conv_part + Y) // 2.0
    
    max_score = np.max(scores)
    best_idx = np.argmax(scores)
    best_N = N_arr[best_idx]
    
    # Check time
    if time.time() > deadline - 10:
         # Not enough time to retrieve pairs? 
         # Retrieving pairs is O(M). M ~ 10^4. Fast.
         pass
        
    # Retrieve pairs for best_N
    p_set = set(np.nonzero(powerful_bool)[0].tolist())
    pairs = []
    # Iterate a from 1 to best_N // 2
    # Check if a in p_set and best_N - a in p_set
    # To speed up, iterate only over powerful numbers a <= best_N // 2.
    
    # Get list of powerful numbers <= best_N // 2
    half_N = best_N // 2
    powerful_list = [x for x in np.nonzero(powerful_bool)[0] if x <= half_N]
    
    for a in powerful_list:
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    # Ensure distinct pairs a <= b
    # The loop ensures a <= b.
    # Are pairs distinct? Yes, because a is unique in the list.
    
    return {'N': int(best_N), 'pairs': pairs}
