"""Rewritten program: find N with many representations as sum of two powerful numbers."""
import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    # A powerful number n has the property that for every prime p | n, p^2 | n.
    # Such numbers can be written as a^2 * b^3 where b is square-free.
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    # Sieve for square-free numbers up to max_b
    is_sq_free = [True] * (max_b + 1)
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        for j in range(sq, max_b + 1, sq):
            is_sq_free[j] = False
    is_sq_free[0] = False
    if max_b >= 1:
        is_sq_free[1] = True  # 1 is square-free

    powerful_set = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b * b * b
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful_set.add(val)
    return sorted(powerful_set)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # We want to maximize the number of pairs (a, b) with a + b = N, a <= b,
    # both powerful, N <= 10^18.
    # Strategy: generate powerful numbers up to some limit L, then use
    # np.bincount on pairwise sums to find N with max count.
    # With M powerful numbers, we have ~M(M+1)/2 pairs.
    # For L = 2*10^8, M ~ 6000, pairs ~ 18M.
    # For L = 10^9, M ~ 9000, pairs ~ 40M.
    # For L = 5*10^9, M ~ 20000, pairs ~ 200M (may be too slow).
    
    # Let's try increasing L and see what fits.
    # Start with L = 10^9
    L = 10**9
    
    # Generate powerful numbers up to L
    pw = _generate_powerful(L)
    M = len(pw)
    
    # We need a + b = N where a, b are powerful, a <= b.
    # N can be up to 2*L.
    # For each pair (i, j) with i <= j, compute pw[i] + pw[j], count occurrences.
    # Use np.bincount for efficiency.
    
    # Convert to numpy array
    pw_arr = np.array(pw, dtype=np.int64)
    
    # Generate all upper-triangular pairs
    # For M ~ 9000, M(M+1)/2 ~ 40M pairs. This might be memory heavy.
    # Let's do it in chunks to be safe.
    
    # Actually, let's first check if M(M+1)/2 is manageable.
    n_pairs = M * (M + 1) // 2
    print(f"Generated {M} powerful numbers, {n_pairs} pairs")
    
    if n_pairs > 100_000_000:
        # Too many, reduce L
        L = 2 * 10**8
        pw = _generate_powerful(L)
        M = len(pw)
        pw_arr = np.array(pw, dtype=np.int64)
        n_pairs = M * (M + 1) // 2
        print(f"Reduced: {M} powerful numbers, {n_pairs} pairs")
    
    # Use bincount on the sums
    # Create all pairs (i, j) with i <= j
    # To save memory, process in chunks of rows
    
    chunk_size = 1000
    max_sum = int(pw_arr[-1]) * 2
    
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        # For each i in [i_start, i_end), j ranges from i to M-1
        # Create array of i values repeated appropriately
        i_vals = np.arange(i_start, i_end, dtype=np.int64)
        # For each i, j goes from i to M-1
        # Total elements: sum_{i=i_start}^{i_end-1} (M - i)
        # We can create the pairs as: for each i, j = i, i+1, ..., M-1
        # Flatten: concatenate arrays
        
        # Build j values: for i_start, j = i_start..M-1; for i_start+1, j = i_start+1..M-1; etc.
        j_vals_list = []
        sum_list = []
        for i in range(i_start, i_end):
            j_range = np.arange(i, M, dtype=np.int64)
            j_vals_list.append(j_range)
            sum_list.append(pw_arr[i] + pw_arr[j_range])
        
        j_flat = np.concatenate(j_vals_list)
        sums_flat = np.concatenate(sum_list)
        
        # Filter: we only care about N <= 10^18, but since L <= 10^9, N <= 2*10^9, so fine.
        # Also a <= b is guaranteed since i <= j and pw is sorted.
        # Count occurrences
        # Use np.bincount for this chunk and add to global counts
        chunk_counts = np.bincount(sums_flat.astype(np.int64), minlength=max_sum + 1)
        counts += chunk_counts.astype(np.int32)
        
        if time.time() > deadline:
            break
    
    # Find the N with maximum count
    # N must be >= 2 (minimum sum is 1+1=2)
    best_N = int(np.argmax(counts[2:])) + 2
    best_count = int(counts[best_N])
    
    print(f"Best N = {best_N}, count = {best_count}")
    
    # Now verify and build the pairs list
    pw_set = set(pw)
    pairs = []
    for a in pw:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in pw_set and a <= b:
            pairs.append([a, b])
    
    # Verify: each pair should have a + b = best_N, both powerful
    # The score is len(pairs)
    
    return {'N': best_N, 'pairs': pairs}
