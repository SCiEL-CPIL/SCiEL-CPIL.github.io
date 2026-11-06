"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    if limit < 1:
        return [1]
    powerful = set()
    powerful.add(1)
    
    # b^3 * a^2 <= limit
    # b up to limit^(1/3)
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # a^2 <= limit / b3
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    best_N = 2
    best_pairs = [[1, 1]]
    
    # Try different FFT sizes (powers of 2) to balance memory and search range
    # Size 2^24 is ~16M, 2^25 is ~33M, 2^26 is ~67M
    # We want to search as high as possible.
    # Let's try 2^25 first.
    
    fft_sizes = [2**24, 2**25, 2**26]
    
    for size in fft_sizes:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
            
        # The max N we can represent is size - 1
        # But we only need powerful numbers up to size // 2 roughly?
        # No, if a+b=N, and N < size, then a < size and b < size.
        # So we need powerful numbers up to size.
        # However, if a is close to size, b must be small.
        # Generating powerful numbers up to `size` is O(sqrt(size)).
        
        max_powerful_limit = size
        powerful = _generate_powerful(max_powerful_limit)
        powerful_set = set(powerful)
        
        if not powerful:
            continue
            
        # Create indicator array
        A = np.zeros(size, dtype=np.float64)
        for p in powerful:
            if p < size:
                A[p] = 1.0
        
        # Perform FFT
        F = np.fft.fft(A)
        F2 = F * F
        C = np.fft.ifft(F2).real
        
        # C[n] is the number of ordered pairs (a,b) such that a+b=n
        # We want pairs a <= b.
        # Count = (C[n] + 1_{n even and n/2 powerful}) / 2
        
        # We are interested in N >= 2
        # Find max count
        # To avoid memory issues with huge arrays, we can just find the argmax
        # But we need to handle the a=b case for the count calculation later?
        # Actually, for finding the best N, we can just look at C.
        # C is proportional to the number of pairs.
        
        # Exclude indices 0 and 1
        if len(C) > 2:
            # Find the index with max value in range [2, size)
            # Note: C might have small numerical errors, but max should be fine.
            max_idx = int(np.argmax(C[2:]) + 2)
            
            # Calculate actual number of pairs for this N
            N = max_idx
            count_ordered = int(round(C[N]))
            
            # Adjust for a <= b
            # If N is even, check if N/2 is powerful
            if N % 2 == 0:
                half = N // 2
                if half in powerful_set:
                    # One pair is (half, half). The rest are (a, b) with a != b.
                    # Ordered pairs: 1 + 2 * (num pairs with a < b)
                    # So num pairs with a < b = (count_ordered - 1) / 2
                    # Total pairs a <= b = 1 + (count_ordered - 1) / 2 = (count_ordered + 1) / 2
                    num_pairs = (count_ordered + 1) // 2
                else:
                    # No a=b.
                    # Ordered pairs: 2 * (num pairs with a < b)
                    # num pairs with a < b = count_ordered / 2
                    # Total pairs a <= b = count_ordered / 2
                    num_pairs = count_ordered // 2
            else:
                # N is odd, so a != b always.
                num_pairs = count_ordered // 2
            
            if num_pairs > len(best_pairs):
                best_N = N
                # Extract pairs
                best_pairs = []
                for a in powerful:
                    if 2 * a > N:
                        break
                    b = N - a
                    if b in powerful_set and a <= b:
                        best_pairs.append([a, b])
                
                # Verify count matches
                if len(best_pairs) != num_pairs:
                     # Recount just in case
                     best_pairs = []
                     for a in powerful:
                         if 2 * a > N:
                             break
                         b = N - a
                         if b in powerful_set and a <= b:
                             best_pairs.append([a, b])
    
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
