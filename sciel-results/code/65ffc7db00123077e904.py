import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    timeout = 145  # safety margin before 150s limit
    
    # Target M - increased to explore larger N for more representations
    M = 200_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            if p <= M:
                powerful.add(p)
    
    powerful_list = sorted(powerful)
    powerful_set = set(powerful_list)
    n_list = len(powerful_list)
    
    if n_list < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Use FFT convolution for fast pairwise sum counting
    max_sum = 2 * M
    
    # Create indicator array
    indicator = np.zeros(max_sum + 1, dtype=np.float64)
    for p in powerful_list:
        indicator[p] = 1.0
    
    # FFT convolution: sum of two powerful numbers
    # We need N = a + b where a, b are powerful, a <= b
    # Convolution of indicator with itself gives count of ordered pairs
    # For a != b, each unordered pair {a,b} with a<b contributes 2 to the convolution
    # For a = b, it contributes 1
    # So: number of unordered pairs = (conv[N] + (1 if N is twice a powerful)) / 2
    # But we want pairs with a <= b, so:
    # count_unordered = (conv[N] + indicator[N//2] if N even else conv[N]) / 2
    
    # Use FFT for convolution
    # Size needs to be at least 2*max_sum + 1 for full convolution
    fft_size = 1
    while fft_size < 2 * (max_sum + 1):
        fft_size *= 2
    
    if time.time() - start_time > timeout - 30:
        # Fallback to simpler approach if we're running out of time
        return {'N': 2, 'pairs': [[1, 1]]}
    
    fft_indicator = np.fft.rfft(indicator, n=fft_size)
    fft_conv = np.fft.irfft(fft_indicator * fft_indicator, n=fft_size)
    
    # The convolution gives us counts of ordered pairs (a,b) where a+b=N
    # For a < b: counted twice (as (a,b) and (b,a))
    # For a = b: counted once
    # We want unordered pairs with a <= b
    # So: unordered_count[N] = (fft_conv[N] + (1 if N%2==0 and (N//2) in powerful_set else 0)) / 2
    # But actually, since we're using 0/1 indicators:
    # conv[N] = sum over all a: indicator[a] * indicator[N-a]
    # This counts each ordered pair (a, N-a) where both are powerful
    # For a != N-a (i.e., N even and a != N/2): pairs (a,N-a) and (N-a,a) both counted
    # For a = N-a (i.e., N even and a = N/2): only one term
    # So number of unordered pairs with a <= b is:
    # If N is odd: conv[N] / 2 (all pairs have a != b)
    # If N is even: (conv[N] - 1) / 2 + 1 if N/2 is powerful, else (conv[N]) / 2
    # Wait, let me reconsider.
    
    # conv[N] = number of (a, b) with a + b = N, a and b in powerful, counting order
    # We want pairs [a, b] with a <= b, a + b = N
    # Each pair {a, b} with a < b contributes 2 to conv[N]
    # Each pair {a, a} with 2a = N contributes 1 to conv[N]
    # So: num_pairs = (conv[N] + num_self_pairs) / 2 where num_self_pairs = 1 if N even and N/2 in powerful else 0
    # Actually: num_pairs = conv[N] / 2 if no self-pair, or (conv[N] - 1)/2 + 1 if self-pair exists
    # Simplified: num_pairs = ceil(conv[N] / 2) if there's a self-pair, else conv[N] / 2
    # More precisely: 
    #   Let s = 1 if (N % 2 == 0 and (N//2) in powerful_set) else 0
    #   num_pairs = (conv[N] + s) / 2
    
    # Find the N with maximum number of pairs
    # We need to compute (conv[N] + s_N) / 2 for each N and find the max
    
    # Create the self-pair indicator array
    self_pair = np.zeros(max_sum + 1, dtype=np.float64)
    for p in powerful_list:
        if 2 * p <= max_sum:
            self_pair[2 * p] = 1.0
    
    # Compute the score for each N: (conv[N] + self_pair[N]) / 2
    scores = (fft_conv[:max_sum + 1] + self_pair) / 2.0
    
    # Find the best N
    best_N = int(np.argmax(scores))
    best_count = int(round(scores[best_N]))
    
    if best_count < 1:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Reconstruct pairs
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
    
    # Verify we found the right number of pairs
    if len(pairs) < best_count:
        # This shouldn't happen, but just in case
        best_count = len(pairs)
    
    return {'N': best_N, 'pairs': pairs}
