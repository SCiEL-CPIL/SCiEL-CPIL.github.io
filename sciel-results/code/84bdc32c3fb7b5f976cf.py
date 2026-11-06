import time
import numpy as np
from math import isqrt, gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 10s safety margin
    deadline = start_time + 140.0
    
    def generate_powerful(limit):
        """
        Generate all powerful numbers <= limit.
        A number is powerful if it is of the form u^2 * v^3.
        """
        pw = set()
        # v^3 <= limit => v <= limit^(1/3)
        max_v = int(limit ** (1/3)) + 2
        
        for v in range(1, max_v + 2):
            v_cubed = v ** 3
            if v_cubed > limit:
                break
            # u^2 <= limit / v^3 => u <= sqrt(limit / v^3)
            max_u = isqrt(limit // v_cubed)
            for u in range(1, max_u + 1):
                p = u * u * v_cubed
                if p <= limit:
                    pw.add(p)
        return sorted(pw)

    # We want to find N with many representations a+b=N where a,b are powerful.
    # The "Parameterized Search" strategy suggests that N can be constructed
    # as the sum of two "core" structures, and then we verify how many other
    # powerful numbers can form N.
    
    # However, a more direct structural insight is:
    # If N is a sum of two powerful numbers, and we want many representations,
    # N should be "rich" in powerful number divisors/summands.
    # 
    # Let's use the insight from "inspiration 2": 
    # Enumerate pairs of small parameters (u1, v1) and (u2, v2) to generate
    # candidate N values, then count representations.
    # 
    # Actually, a simpler and more robust structural change is:
    # Instead of FFT on a dense array (which is memory heavy and slow for large M),
    # use a "Bucketed Sparse Convolution" or simply "Exhaustive Pair Counting"
    # on a moderately large set of powerful numbers, but optimized with NumPy.
    # 
    # The previous program used FFT for M=20M. 
    # Let's try a different structural approach:
    # Generate powerful numbers up to M=100,000,000.
    # The number of such numbers K is approx 100M^(1/2) * constant? 
    # Count of powerful numbers <= x is ~ C * sqrt(x).
    # For x=10^8, sqrt(x)=10^4. C is around 1.3. So K ~ 13,000.
    # Pair counting is O(K^2) = 169,000,000 operations.
    # This is too slow in pure Python, but can be vectorized or done in C.
    # Since we only have Python, O(K^2) is ~10^8 ops, which is ~10-30s in Python.
    # This is feasible within 140s!
    
    # Strategy:
    # 1. Generate powerful numbers up to M = 100,000,000.
    # 2. Compute all pair sums (i, j) with i <= j.
    # 3. Count occurrences of each sum N.
    # 4. Find N with max count.
    # 5. Return N and the pairs.
    
    # To speed up pair counting:
    # We can use NumPy to generate sums in chunks or use a dictionary.
    # A dictionary of size K^2 is too big.
    # But we only need the MAX count.
    # We can use a "Meet in the Middle" or just iterate and count.
    
    # Let's use a dictionary for counts.
    # K ~ 13,000.
    # Pairs ~ 85,000,000.
    # Python loop: 85M iterations. Might be too slow (10-20s).
    # Let's try to optimize with NumPy if possible, or just use a tight loop.
    
    M = 100_000_000
    
    # Check time
    if time.time() > deadline - 30:
        M = 50_000_000
        
    powerful_list = generate_powerful(M)
    K = len(powerful_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    powerful_set = set(powerful_list)
    
    # If K is small enough, do exhaustive counting.
    # Threshold: K < 20,000 => Pairs < 200M.
    # Let's try it.
    
    counts = defaultdict(int)
    # To make it faster, we can break early if we find a very high count?
    # No, we need the max.
    
    # Optimization: 
    # Instead of iterating i from 0 to K-1 and j from i to K-1,
    # we can just do it.
    
    # Let's use a list for faster access?
    # pw = powerful_list
    # for i in range(K):
    #   ai = pw[i]
    #   for j in range(i, K):
    #     counts[ai + pw[j]] += 1
    
    # This is O(K^2).
    # Let's check if K is reasonable.
    # For M=10^8, K is likely around 10,000-15,000.
    # 15,000^2 / 2 = 112,500,000.
    # In Python, 100M simple ops take ~5-10 seconds.
    # This is acceptable.
    
    # Let's run the counting.
    # We can potentially parallelize? No, GIL.
    # But we can use NumPy to generate sums?
    # np.add.outer(pw, pw) creates a KxK matrix.
    # 15000x15000 = 225M elements. 225M * 8 bytes = 1.8 GB. Too much memory.
    # So we must use the loop or chunked NumPy.
    
    # Chunked NumPy approach:
    # Process i in chunks of size C.
    # For each chunk, compute sums with all j >= i?
    # Actually, we just need to count all pairs (i,j) with i<=j.
    # We can compute the full matrix in chunks of rows.
    
    chunk_size = 1000
    counts_dict = defaultdict(int)
    
    # Pre-convert to numpy array for faster access in inner loop?
    # Or just use list.
    pw = powerful_list
    
    # If K is very large, this might be slow.
    # Let's add a time check inside the loop?
    # No, that's inefficient.
    
    # Let's just run it.
    for i in range(K):
        if time.time() > deadline - 20:
            break
        ai = pw[i]
        # We can vectorize the inner loop?
        # sums = ai + pw[i:]
        # Then count unique values in sums?
        # np.unique(sums, return_counts=True)
        # This is O(K log K) per i. Total O(K^2 log K).
        # This is slower than O(K^2) but uses C-level loops.
        # Let's try this.
        
        if K - i > 100: # Only vectorize if chunk is large enough
            sums = ai + np.array(pw[i:])
            unique_sums, unique_counts = np.unique(sums, return_counts=True)
            for u_s, u_c in zip(unique_sums, unique_counts):
                counts_dict[u_s] += u_c
        else:
            for j in range(i, K):
                counts_dict[ai + pw[j]] += 1
                
    if not counts_dict:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find max count
    best_N, best_count = max(counts_dict.items(), key=lambda x: x[1])
    
    # Retrieve pairs for best_N
    best_pairs = []
    for a in pw:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            best_pairs.append([a, b])
            
    return {'N': best_N, 'pairs': best_pairs}
