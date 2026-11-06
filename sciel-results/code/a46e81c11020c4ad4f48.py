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
    
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def _is_powerful(n):
    """Check if n is powerful using exact factorization."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count == 1:
                return False
        p += 1
    if temp > 1:
        return False  # remaining prime factor with exponent 1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    # Try increasing limits
    # We can now go higher due to memory-efficient counting
    limits_to_try = [10**8, 5*10**8, 10**9]
    
    for limit in limits_to_try:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        powerful = _generate_powerful(limit)
        powerful_set = set(powerful)
        arr = np.array(powerful, dtype=np.int64)
        M = len(arr)
        
        # Estimate max N
        max_N = 2 * limit
        
        # Allocate counter array
        # Using uint16 to save memory. Max count is bounded by M (~30k), so uint16 (max 65535) is safe.
        try:
            counts = np.zeros(max_N + 1, dtype=np.uint16)
        except MemoryError:
            # If memory fails, try a smaller limit or break
            continue

        # Accumulate counts
        # Iterate over each powerful number a
        # For each a, add a + b for all b >= a
        for i in range(M):
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            
            ai = arr[i]
            # Generate sums ai + arr[i:]
            # This is a view or small array
            sums = ai + arr[i:]
            
            # Count occurrences of each sum in this row
            # np.unique is efficient for small arrays
            unique_sums, local_counts = np.unique(sums, return_counts=True)
            
            # Accumulate into global counts
            # unique_sums are distinct, so direct indexing is safe and fast
            counts[unique_sums] += local_counts
        
        if time.time() - start_time > deadline - 10:
            continue

        # Find the best N
        # We only care about N >= 2
        # max count
        best_count_here = int(np.max(counts[2:]))
        if best_count_here > 0:
            best_N = int(np.argmax(counts[2:]) + 2)
            
            if best_count_here > best_count:
                best_count = best_count_here
                pairs = []
                for a in powerful:
                    if 2 * a > best_N:
                        break
                    b = best_N - a
                    if b in powerful_set and a <= b:
                        pairs.append([a, b])
                best_result = {'N': best_N, 'pairs': pairs}
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
    
    return best_result

# EVOLVE-BLOCK-END
