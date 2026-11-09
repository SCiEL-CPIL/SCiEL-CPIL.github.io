import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    # Safety margin: 110s
    deadline = start_time + 110.0
    
    # Choose L such that M (count of powerful numbers) allows block-based NumPy counting
    # to finish within time.
    # L = 10^11 -> M ~ 3e5. Blocks of 1000 -> 300 blocks. ~45k block pairs.
    # Each block pair sum + unique is fast.
    L = 10**11
    
    # Generate powerful numbers up to L
    # Powerful numbers are x^2 * y^3 where y is square-free.
    # Max y is L^(1/3).
    # Max x is sqrt(L / y^3).
    
    limit_primes = int(L ** (1.0/3.0)) + 2
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_primes ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def is_square_free(n):
        if n < 2:
            return True
        for p in primes:
            if p * p > n:
                break
            if n % (p * p) == 0:
                return False
        return True
        
    powerful_set = set()
    max_b = int(L ** (1.0/3.0))
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= L:
                powerful_set.add(val)
            
    powerful_list = sorted(powerful_set)
    M = len(powerful_list)
    
    if M == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # If M is small enough, we can try a larger L? 
    # For now, proceed with block counting.
    
    p_array = np.array(powerful_list, dtype=np.int64)
    
    # Block size for NumPy operations
    # We want B^2 to be manageable (e.g., 10^6 elements)
    B = 1000
    num_blocks = (M + B - 1) // B
    
    counts = defaultdict(int)
    
    # Optimization: We only need i <= j for distinct pairs a<=b?
    # The task asks for distinct pairs a <= b.
    # If we count all (i,j), we get 2x for i!=j and 1x for i==j.
    # We can just count all and divide? No, because a=b is possible.
    # Easier: iterate blocks i from 0 to num_blocks-1
    #         iterate blocks j from i to num_blocks-1
    #         compute sums for p_array[i*B:(i+1)*B] + p_array[j*B:(j+1)*B]
    #         if i == j, we must ensure a <= b. Since the array is sorted, 
    #         within the same block, p[k] + p[l] with k<=l corresponds to a<=b.
    #         However, np broadcasting computes all pairs.
    #         We can filter later or just count all and adjust?
    #         Actually, the problem says "distinct pairs a <= b".
    #         If we count all pairs (i,j) in the full matrix, we count (a,b) and (b,a) for a!=b.
    #         The score is the number of pairs a<=b.
    #         So if we count all sums, the count for N will be:
    #         2 * (number of pairs a<b) + (number of pairs a=b).
    #         We want (number of pairs a<b) + (number of pairs a=b).
    #         Let C_total be the count from full matrix.
    #         Let C_diag be the count where a=b (i.e., 2a=N).
    #         Then desired count = (C_total + C_diag) / 2.
        
    # Let's just accumulate counts for all pairs (i,j) including i>j?
    # No, let's stick to i <= j blocks to save half the time.
    # For block i and block j (i < j): all a in block i are <= all b in block j?
    # Yes, because the list is sorted. So all a in block i < all b in block j.
    # So for i < j, all pairs satisfy a < b.
    # For i == j, we need a <= b.
    
    for i in range(num_blocks):
        if time.time() > deadline:
            break
        block_i = p_array[i*B : (i+1)*B]
        
        for j in range(i, num_blocks):
            if time.time() > deadline:
                break
            block_j = p_array[j*B : (j+1)*B]
            
            # Compute sums
            # To avoid huge memory, we can process in sub-chunks if B is large, 
            # but B=1000 -> 10^6 sums is ~8MB. Fine.
            sums = block_i[:, None] + block_j[None, :]
            
            # Count occurrences
            # np.unique is efficient
            unique_sums, counts_arr = np.unique(sums, return_counts=True)
            
            for val, cnt in zip(unique_sums, counts_arr):
                counts[val] += cnt
                
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the N with max count
    # Note: The counts accumulated are for pairs (a,b) where a is in block i and b in block j.
    # For i < j, a < b always.
    # For i == j, we counted ALL pairs in the block, including a > b.
    # We need to correct for the diagonal blocks (i == j).
    
    # Let's re-evaluate the counting logic.
    # It is safer to just count all pairs (i,j) for i from 0..M, j from 0..M? No, too slow.
    
    # Let's assume the counts above are:
    # For i < j: counts all pairs (a,b) with a in i, b in j. Since sorted, a < b.
    # For i == j: counts all pairs (a,b) in the same block. This includes a < b, a = b, a > b.
    # We want to count only a <= b.
    # For i < j, the count is correct (all are a < b).
    # For i == j, we have double counted the a < b part? No, we just have all pairs.
    # Let C_i be the count accumulated for block i with itself.
    # C_i includes:
    # - pairs (a,b) with a < b
    # - pairs (a,b) with a = b
    # - pairs (a,b) with a > b
    # Since the set of pairs is symmetric, count(a<b) = count(a>b).
    # Let K be count(a=b).
    # C_i = 2 * count(a<b) + K.
    # We want count(a<=b) = count(a<b) + K = (C_i - K) / 2 + K = (C_i + K) / 2.
    
    # So we need to calculate K for each diagonal block and adjust the total count for N.
    
    # However, we accumulated counts in a single dictionary `counts`.
    # We need to know which counts came from diagonal blocks and their K values.
    
    # Let's redo the accumulation to separate diagonal and off-diagonal.
    # Or, simpler: 
    # Total Count for N = Sum(OffDiag) + Sum(Diag_Corrected)
    
    # Let's restart the counting logic to be precise.
    
    counts_off = defaultdict(int)
    counts_diag_raw = defaultdict(int)
    diag_diag_counts = defaultdict(int) # K for each N in diagonal blocks
    
    for i in range(num_blocks):
        if time.time() > deadline:
            break
        block_i = p_array[i*B : (i+1)*B]
        
        # Diagonal block
        if i == i: # Always true, just for logic clarity
            sums_diag = block_i[:, None] + block_i[None, :]
            # Count a=b cases
            # a=b implies 2a=N. 
            # In the block, a=b occurs when the value appears twice? No, a and b are from the same list.
            # a=b means we pick the same element twice? 
            # The pair is (a, b) where a and b are values from the list.
            # If we pick index k and l, a=p[k], b=p[l].
            # a=b iff p[k]=p[l]. Since the list has distinct elements (it's a set), 
            # a=b iff k=l.
            # So for each element x in block_i, the pair (x,x) contributes 1 to K for N=2x.
            
            # We can just count how many times each N appears in sums_diag.
            # And separately count how many x in block_i satisfy 2x = N.
            
            unique_sums_diag, cnts_diag = np.unique(sums_diag, return_counts=True)
            for val, cnt in zip(unique_sums_diag, cnts_diag):
                counts_diag_raw[val] += cnt
            
            # Count K (diagonal elements)
            # For each x in block_i, N = 2x is a diagonal pair.
            # There is exactly 1 such pair for each x.
            for x in block_i:
                N_diag = 2 * x
                diag_diag_counts[N_diag] += 1

        # Off-diagonal blocks j > i
        for j in range(i + 1, num_blocks):
            if time.time() > deadline:
                break
            block_j = p_array[j*B : (j+1)*B]
            sums_off = block_i[:, None] + block_j[None, :]
            unique_sums_off, cnts_off = np.unique(sums_off, return_counts=True)
            for val, cnt in zip(unique_sums_off, cnts_off):
                counts_off[val] += cnt

    # Combine
    final_counts = defaultdict(int)
    
    # Off-diagonal counts are already correct (a < b)
    for N, c in counts_off.items():
        final_counts[N] += c
        
    # Diagonal counts need correction
    # Raw count C_i for N in block i.
    # Corrected = (C_i + K_N) / 2
    for N, c in counts_diag_raw.items():
        K = diag_diag_counts.get(N, 0)
        # c is the raw count of pairs (a,b) in the same block summing to N.
        # K is the number of pairs (a,a) summing to N.
        # Note: K is part of c.
        # c = 2 * count(a<b) + K
        # desired = count(a<b) + K = (c - K) / 2 + K = (c + K) / 2
        # Since c and K have the same parity? 
        # c - K is even. So (c+K)/2 is an integer.
        corrected = (c + K) // 2
        final_counts[N] += corrected

    if not final_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(final_counts, key=final_counts.get)
    best_count = final_counts[best_N]
    
    # Retrieve pairs
    p_set = set(powerful_list)
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
