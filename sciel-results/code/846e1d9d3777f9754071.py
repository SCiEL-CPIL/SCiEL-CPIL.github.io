"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Strategy: Generate powerful numbers up to a limit L.
    # Then, compute all pairwise sums a+b (with a<=b) and count frequencies.
    # The N with the highest frequency has the most representations.
    # We need to ensure N <= 10^18. If L is 10^8, max sum is 2*10^8, which is safe.
    # We can go higher with L. Let's try L = 10^9.
    # Number of powerful numbers up to 10^9 is roughly ~20,000-30,000?
    # Let's estimate: 
    # Powerful numbers are of form a^2 * b^3.
    # For b=1, a^2 <= 10^9 => a <= 31622.
    # For b=2, 8a^2 <= 10^9 => a <= 11180.
    # The count is dominated by small b.
    # Total count is likely around 10^4 to 10^5.
    # Pairwise sums of 10^5 numbers is 10^10 operations, which is too slow in Python.
    # We need a smarter way or a smaller L.
    
    # Let's optimize the generation and summing.
    # If M is the count of powerful numbers, summing all pairs is O(M^2).
    # If M = 10,000, M^2 = 10^8. In Python, 10^8 simple operations take ~10-30s.
    # This is risky.
    
    # Alternative: Use FFT? 
    # We can create a binary array A where A[x] = 1 if x is powerful, else 0.
    # The number of representations of N as a+b is the coefficient of x^N in A(x)^2.
    # We can compute this using FFT (numpy.fft).
    # Length of array needs to be > max_sum.
    # If L = 10^8, max_sum = 2*10^8. Array size 2^28 = 268 million.
    # FFT of size 2^28 is fast in numpy (C implementation).
    # Memory: 268M floats ~ 2GB. This might be too much for standard eval environments?
    # Let's be conservative. L = 10^7. Max sum 2*10^7.
    # Next power of 2 is 2^25 = 33M.
    # Memory: 33M * 8 bytes = 264 MB. Safe.
    # Time: FFT of 33M is very fast (< 1s).
    # Count of powerful numbers up to 10^7:
    # b=1: a<=3162 (3162 nums)
    # b=2: a<=1118 (1118 nums)
    # ...
    # Total M is likely ~5000-10000.
    # This is a very different approach: Spectral counting vs Direct enumeration.
    
    L = 10**7
    # Generate powerful numbers up to L
    powerfuls = set()
    max_b = int(L ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= L:
                powerfuls.add(val)
    
    # Include 1? 1 is powerful (1^2 * 1^3). The loop covers b=1, a=1 -> 1.
    # The set contains unique powerful numbers.
    
    # Prepare array for FFT
    # We want to find N such that count of (a,b) in powerfuls with a+b=N is max.
    # Note: The problem asks for distinct pairs [a,b] with a<=b.
    # FFT convolution gives ordered pairs (a,b) and (b,a) and (a,a).
    # Let C(N) be the number of ordered pairs (a,b) such that a+b=N.
    # If a != b, the pair {a,b} contributes 2 to C(N).
    # If a == b, the pair {a,a} contributes 1 to C(N).
    # We want to maximize the number of distinct pairs.
    # Let k be the number of distinct pairs.
    # Let d be the number of diagonal pairs (a=a) i.e., N is even and N/2 is powerful.
    # Then C(N) = 2 * (k - d) + d = 2k - d.
    # So k = (C(N) + d) / 2.
    # To maximize k, we should look for N with high C(N), and prefer those where d is large?
    # Actually, for a fixed C(N), larger d means smaller k?
    # Wait. C(N) = sum_{a+b=N} 1.
    # If N is even and N/2 is powerful, one term is a=N/2, b=N/2.
    # All other terms come in pairs (a,b) and (b,a).
    # So C(N) is odd if N/2 is powerful, even otherwise.
    # k = (C(N) + (1 if N/2 in powerfuls else 0)) / 2.
    
    # We can just compute C(N) via FFT, then calculate k for each candidate N.
    # We don't need to check all N, just the top few candidates from C(N).
    
    max_sum = 2 * L
    # Next power of 2 for FFT
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
    
    # Create frequency array
    freq = [0] * n_fft
    for p in powerfuls:
        freq[p] = 1
        
    import numpy as np
    freq_np = np.array(freq, dtype=np.float64)
    
    # Convolution
    fft_freq = np.fft.fft(freq_np)
    conv = np.fft.ifft(fft_freq * fft_freq)
    conv = np.real(conv)
    
    # conv[i] is the number of ordered pairs summing to i.
    # We want to find N in [2, 10^18] (but our sums are <= 2*L) with max pairs.
    # Since we are limited to L, N <= 2*L.
    # 2*L = 2*10^7 < 10^18. So all found N are valid.
    
    # Get top candidates
    # We need to be careful with floating point errors in FFT.
    # conv values should be close to integers.
    
    # Find indices with highest values
    # We can't sort 33M elements easily? Actually we can, but it's O(N log N).
    # Better: Use np.argpartition or just check the top K.
    # Let's get the top 100 candidates.
    
    # Note: We must ensure N >= 2.
    # Also, we should probably restrict N to be within the range of our generated powerfuls sums.
    
    # Get indices of top 1000 values
    # To avoid sorting the whole array, we can use a heap or just take the max and remove?
    # Or simply: since we expect the max to be relatively high, we can threshold.
    # But let's just find the global max and its neighbors.
    
    # Efficient way: 
    # 1. Find max value.
    # 2. Find all indices with value close to max?
    # Or just take the top 50 indices using argpartition.
    
    # argpartition is O(N).
    top_indices = np.argpartition(conv, -50)[-50:]
    
    best_N = -1
    best_pairs = []
    best_score = -1
    
    powerfuls_set = set(powerfuls) # For quick lookup
    
    for idx in top_indices:
        N = int(idx)
        if N < 2:
            continue
            
        # Calculate exact number of pairs for this N
        # We need to count a in powerfuls such that N-a in powerfuls and a <= N-a.
        # This is O(M) for each candidate. 50 * 10000 = 500,000 ops. Fast.
        
        count = 0
        pairs = []
        # Iterate through powerfuls
        # Optimization: only iterate a <= N/2
        limit_a = N // 2
        for a in powerfuls:
            if a > limit_a:
                break # Since powerfuls is a set, we can't break easily unless sorted.
            # Let's sort powerfuls first? 
            # We generated them unsorted. Let's sort once.
        
    # Let's sort powerfuls first to allow early termination in the loop above.
    powerfuls_sorted = sorted(powerfuls)
    
    for idx in top_indices:
        N = int(idx)
        if N < 2:
            continue
            
        count = 0
        pairs = []
        for a in powerfuls_sorted:
            if a > N // 2:
                break
            b = N - a
            if b in powerfuls_set:
                count += 1
                pairs.append([a, b])
                
        if count > best_score:
            best_score = count
            best_N = N
            best_pairs = pairs
            
        # Time check
        if time.time() > deadline - 5:
            break
            
    if best_N == -1:
        # Fallback to a simple powerful sum if FFT failed or found nothing
        # e.g., 1+1=2
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
