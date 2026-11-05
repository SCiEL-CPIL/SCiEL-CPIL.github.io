import time
import numpy as np
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Parameters:
    # We want to maximize N up to 10^18, but practically limited by generation time.
    # Strategy: 
    # 1. Generate powerful numbers up to M_LARGE (e.g., 10^8 or 2*10^8).
    # 2. Split into Small (<= M_SMALL) and Large (> M_SMALL).
    # 3. Use FFT on Small set to find top candidates for N in [2, 2*M_SMALL].
    #    Wait, if we only look at N in [2, 2*M_SMALL], we miss N where one part is Large.
    #    Actually, if N = a + b, and we assume a, b are powerful.
    #    If we generate all powerful numbers up to M_MAX, we can just do O(K^2) if K is small enough.
    #    K(M=10^8) is approx 20,000-30,000? 
    #    Let's estimate K. Number of powerful numbers <= X is ~ sqrt(X) + X^(1/3) ... 
    #    For X=10^8, K is around 20,000.
    #    O(K^2) = 4 * 10^8 operations. In Python, this is too slow (approx 40-80s).
    #    In NumPy, we can vectorize the check for a specific N, but not for all N at once easily without huge memory.
    
    # Revised Strategy:
    # 1. Generate powerful numbers up to M_MAX = 2 * 10^7 (K ~ 6000).
    #    This allows O(K^2) to be fast (36 * 10^6 ops), but N is limited to 4 * 10^7.
    #    The current program uses FFT to get N up to 2*M.
    #    If we increase M to 10^8, K ~ 20,000.
    #    We can't do O(K^2) in Python.
    #    
    #    New Idea: "Asymmetric" Search.
    #    Generate powerful numbers up to M_MAX = 5 * 10^7.
    #    Split into A (<= M_A) and B (> M_A).
    #    Let M_A = 10^7. M_B = 5 * 10^7.
    #    Count pairs (a, b) such that a in A, b in B, a+b=N.
    #    This covers N in [M_A+min(B), M_A+max(B)].
    #    Also count pairs (a, b) with a,b in A using FFT (N in [2, 2*M_A]).
    #    Also count pairs (a, b) with a,b in B? No, B is sparse.
    #    
    #    Actually, the most efficient way to find *any* N with many representations 
    #    among a set S is to compute the convolution. 
    #    If we can't fit the convolution in memory, we can use a "sliding window" or 
    #    simply iterate over the smaller set.
    
    # Let's try to push M_MAX as high as possible for generation.
    # Generation of powerful numbers up to 10^8:
    # Loop v up to 10^8^(1/3) ~ 464.
    # Loop u up to isqrt(10^8 / v^3).
    # Total iterations is roughly sum_{v} sqrt(10^8/v^3) ~ 10^4 * sum v^-1.5 ~ small.
    # So generation is fast (< 1s).
    
    M_MAX = 100_000_000  # 10^8
    
    # 1. Generate Powerful Numbers
    powerful_set = set()
    max_v = int(M_MAX ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M_MAX:
            break
        max_u = isqrt(M_MAX // v_cubed)
        for u in range(1, max_u + 1):
            p = u * u * v_cubed
            powerful_set.add(p)
            
    powerful_list = sorted(powerful_set)
    K = len(powerful_list)
    
    # Check time
    if time.time() > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}

    # 2. Strategy:
    #    We want to find N with max representations.
    #    If K is small enough (< 10,000), we can do O(K^2) with NumPy vectorization?
    #    No, O(K^2) creates a KxK matrix of sums, which is K^2 memory.
    #    If K=20,000, K^2 = 4*10^8. 4*10^8 * 8 bytes = 3.2 GB. Too much memory.
    
    #    So we must use a method that doesn't store all pairs.
    #    Method: Iterate over N? No, N is huge.
    #    Method: Iterate over a in powerful_list, and for each a, we want to count how many b in powerful_list make a+b=N for some N?
    #    That's just the convolution.
    
    #    Alternative: Use FFT on a subset, and manual check for the rest?
    #    Let's split the list into chunks?
    
    #    Actually, let's look at the score.
    #    Score 4.0667 came from M=16M (K~5500) using FFT.
    #    If we use M=100M (K~20000), we can't use FFT directly on the whole list.
    #    
    #    However, we can use a "Top-K" heuristic.
    #    The most frequent sums usually involve smaller numbers.
    #    Let's split powerful_list into Small (first K/2) and Large (rest).
    #    1. Compute FFT for Small * Small. Get top candidates N_small.
    #    2. For each N_small, verify count by checking all b in powerful_list (including Large).
    #       Wait, if N is determined by Small*Small, it's likely that the optimal N involves Small numbers.
    #       But what if the optimal N involves one Small and one Large?
    #       The sum would be larger.
    #       
    #    Let's try a different split:
    #    Set A: powerful numbers <= M_A
    #    Set B: powerful numbers in (M_A, M_MAX]
    #    
    #    We compute:
    #    1. Conv(A, A) via FFT. Gives counts for N up to 2*M_A.
    #    2. Conv(A, B) via manual iteration?
    #       For each a in A, for each b in B? No, |A|*|B| is large.
    #       Instead, for each b in B, we can check which a in A satisfy a+b = N for candidate Ns?
    #       
    #    Better: 
    #    Candidate Ns come from Conv(A, A).
    #    Let C be the list of top 1000 Ns from Conv(A, A).
    #    For each N in C, we compute the *exact* count of pairs (a,b) in S x S such that a+b=N.
    #    This is done by iterating a in S (where a <= N/2) and checking if N-a is in S.
    #    Since S is a set, this is O(K) per N.
    #    1000 * 20000 = 20,000,000 operations. This is fast in Python (approx 1-2s).
    #    
    #    Does this find the global max?
    #    It finds the max among Ns that have high density in A+A.
    #    It is possible that an N formed by A+B or B+B has higher count, but is less "dense" in A+A.
    #    However, A contains the smallest powerful numbers, which are the most "frequent" in sums.
    #    B contains larger numbers, which are sparser.
    #    It is highly likely the global max is in A+A or A+B where the A part drives the frequency.
    #    
    #    To be safer, we can also generate candidates from "Binned" convolution of A and B?
    #    No, let's stick to the verification of top A+A candidates.
    #    We can also add a few random candidates from A+B range to check.
    
    M_A = 10_000_000
    A = [x for x in powerful_list if x <= M_A]
    B = [x for x in powerful_list if x > M_A]
    
    # FFT on A
    size = 1
    while size < 2 * M_A + 1:
        size *= 2
        
    arr_A = np.zeros(size, dtype=np.float64)
    for x in A:
        arr_A[x] = 1.0
        
    F = np.fft.rfft(arr_A)
    F_sq = F * F
    counts_float = np.fft.irfft(F_sq, n=size)
    counts_int = np.round(counts_float[:2*M_A + 1]).astype(np.int64)
    
    # Get top 500 candidates from A+A
    top_k = 500
    # Ensure we don't pick indices out of bounds if 2*M_A+1 < size
    valid_range = 2 * M_A + 1
    top_indices = np.argpartition(counts_int[:valid_range], -top_k)[-top_k:]
    
    # Also, let's consider that the best N might be slightly larger, involving B.
    # We can't easily FFT A+B without huge memory.
    # But we can manually check a few "shifted" candidates?
    # No, let's just verify the top 500 from A+A.
    
    best_N = 0
    best_exact_count = 0
    best_pairs = []
    
    powerful_set_local = powerful_set
    
    # Optimization: Pre-compute a list of powerful numbers <= max(top_indices)
    # But max(top_indices) is <= 2*M_A.
    # So we only need to check a <= N/2.
    
    for idx in top_indices:
        N = int(idx)
        if N < 2: continue
        
        count = 0
        pairs = []
        # Iterate a in powerful_list
        # We can break early if a > N/2
        # Since powerful_list is sorted, we can use binary search to find the end index
        # But K is small enough that linear scan with break is fine.
        
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set_local:
                count += 1
                pairs.append([a, b])
                
        if count > best_exact_count:
            best_exact_count = count
            best_N = N
            best_pairs = pairs
            
        if time.time() > deadline - 10:
            break
            
    # Fallback if no pairs found (should not happen)
    if best_exact_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
