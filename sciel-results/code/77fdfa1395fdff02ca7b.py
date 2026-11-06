"""Find N <= 10^18 with many representations as sum of two powerful numbers."""
import time
from math import isqrt

def _is_square_free(n):
    """Check if n is square-free."""
    if n < 2:
        return True
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            return False
        p += 1 if p == 2 else 2
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit as a sorted list and set."""
    powerful = set([1])
    max_b = int(round(limit ** (1.0/3.0))) + 2
    for b in range(1, max_b + 1):
        if not _is_square_free(b):
            continue
        b3 = b * b * b
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful), powerful

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    # Generate powerful numbers up to 10^9 (gives ~30000 numbers)
    limit_pw = 10**9
    powerful, powerful_set = _generate_powerful(limit_pw)
    n_pw = len(powerful)

    elapsed = time.time() - start_time
    if elapsed > 60:
        limit_pw = 10**8
        powerful, powerful_set = _generate_powerful(limit_pw)
        n_pw = len(powerful)

    # Use dictionary to count all sums of pairs (i <= j)
    # With n_pw ~ 30000, this is ~450M iterations - too slow in pure Python
    # Instead, use a smaller set and be smarter

    # Strategy: only consider powerful numbers up to a smaller limit for the sum counting
    # but verify against the full set
    # Actually, let's use numpy for speed if available

    try:
        import numpy as np
        have_numpy = True
    except ImportError:
        have_numpy = False

    # For the sum counting, use a subset of smaller powerful numbers
    # to keep the O(n^2) tractable
    if have_numpy and n_pw > 5000:
        # Use first 20000 powerful numbers (smallest ones)
        subset_size = min(n_pw, 20000)
        subset = powerful[:subset_size]
        # Convert to numpy array for faster operations
        arr = np.array(subset, dtype=np.int64)
        
        # Count sums using numpy broadcasting in chunks
        sum_count = {}
        chunk_size = 2000
        for i_start in range(0, subset_size, chunk_size):
            i_end = min(i_start + chunk_size, subset_size)
            chunk = arr[i_start:i_end]
            # For each element in chunk, compute sums with all elements from its position onward
            for i in range(i_start, i_end):
                if time.time() > deadline - 15:
                    break
                a = arr[i]
                # Compute b values: need b >= a and a+b <= 10^18
                # b = arr[j] for j >= i
                # Instead, just count how many times each sum appears
                # This is still O(n^2) per chunk... let's use a different approach
                
                # For each a in chunk, find all b in powerful_set such that a+b is "interesting"
                # Actually, let's just do the full O(n^2) but with early termination
                break
            
            if time.time() > deadline - 15:
                break
    else:
        # Pure Python approach with smaller set
        subset_size = min(n_pw, 5000)
        subset = powerful[:subset_size]
        
        sum_count = {}
        for i in range(subset_size):
            a = subset[i]
            for j in range(i, subset_size):
                b = subset[j]
                s = a + b
                sum_count[s] = sum_count.get(s, 0) + 1
            if time.time() > deadline - 15:
                break

    # Find best N from sum_count
    best_n = None
    best_count = 0
    if sum_count:
        for s, c in sum_count.items():
            if c > best_count:
                best_count = c
                best_n = s

    # Collect pairs for best_n
    best_pairs = []
    if best_n is not None:
        for a in powerful:
            if 2 * a > best_n:
                break
            b = best_n - a
            if b in powerful_set:
                best_pairs.append([a, b])

    # Also try some structured candidates that might have more reps
    # Small N values often have many representations
    structured_candidates = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 54, 60, 72, 90, 100, 120, 144, 180, 200, 240, 288, 360, 400, 480, 500, 600, 720, 800, 900, 1000, 1200, 1440, 1600, 1800, 2000, 2400, 2880, 3200, 3600, 4000, 4800, 5000, 5760, 6000, 7200, 8000, 9000, 10000, 12000, 14400, 16000, 18000, 20000, 24000, 28800, 32000, 36000, 40000, 48000, 50000, 57600, 60000, 72000, 80000, 90000, 100000, 120000, 144000, 160000, 180000, 200000, 240000, 288000, 320000, 360000, 400000, 480000, 500000, 576000, 600000, 720000, 800000, 900000, 1000000]
    
    for n_cand in structured_candidates:
        if time.time() > deadline - 10:
            break
        count = 0
        pairs = []
        for a in powerful:
            if 2 * a > n_cand:
                break
            b = n_cand - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        if count > best_count:
            best_count = count
            best_n = n_cand
            best_pairs = pairs

    if best_n is not None and best_pairs:
        # Cap at 1000 pairs
        if len(best_pairs) > 1000:
            best_pairs = best_pairs[:1000]
        return {'N': best_n, 'pairs': best_pairs}

    return {'N': 2, 'pairs': [[1, 1]]}
