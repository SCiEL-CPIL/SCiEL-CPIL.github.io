"""
Strategy: Find an N up to 10^18 that is a sum of two powerful numbers in many ways.

1. Generate a large list of powerful numbers up to 10^18.
2. Use batched NumPy operations (np.unique) to count the number of representations for sums of pairs of powerful numbers.
3. Find the N with the maximum number of representations.
4. Verify the result and extract the pairs.
"""

import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    time_limit = 110.0  # 10s margin for 120s limit

    # Step 1: Generate powerful numbers up to 10^18
    # A powerful number is one where every prime factor appears to the power >= 2.
    # We generate them by iterating over squarefree kernel and square.
    # Let n = s * k^2 where s is squarefree. But it's easier to generate by:
    # n = a * b^3 where a is square. Actually, the standard way is:
    # n = prod p_i^{e_i} where e_i >= 2.
    # We can generate by: for each squarefree number s, and for each integer k,
    # n = s * k^2 is powerful iff s=1. No, that's not right.
    # A number is powerful iff it is of the form a * b^2 where a is a square and b is squarefree? No.
    # Let's use the property: n is powerful iff n = m * k^2 where m is a square.
    # Actually, the simplest generation:
    # For each square number a = u^2, and for each cube number c = v^3,
    # n = a * c is powerful.
    # Why? n = u^2 * v^3. The prime factors of u^2 have even exponents (>=2 if they appear in u).
    # The prime factors of v^3 have exponents that are multiples of 3.
    # The combined exponent for a prime p is 2*i + 3*j.
    # If p is in u, i >= 1, so 2*i >= 2. If p is in v, j >= 1, so 3*j >= 3 >= 2.
    # If p is in both, 2*i + 3*j >= 2*1 + 3*1 = 5 >= 2.
    # So all exponents are >= 2.
    # And every powerful number can be written this way?
    # Let n = prod p_i^{e_i} with e_i >= 2.
    # We can write e_i = 2*a_i + 3*b_i.
    # This is possible for all e_i >= 2?
    # 2 = 2*1 + 3*0
    # 3 = 2*0 + 3*1
    # 4 = 2*2 + 3*0
    # 5 = 2*1 + 3*1
    # 6 = 2*3 + 3*0
    # 7 = 2*2 + 3*1
    # Yes, for all e_i >= 2.
    # So n = (prod p_i^{a_i}) * (prod p_i^{b_i})^2 = u^2 * v^2 = (u*v)^2? No.
    # Let's re-express.
    # n = prod p_i^{2*a_i + 3*b_i} = (prod p_i^{a_i})^2 * (prod p_i^{b_i})^3.
    # Let u = prod p_i^{a_i}, v = prod p_i^{b_i}.
    # Then n = u^2 * v^3.
    # So generating all u^2 * v^3 gives all powerful numbers.
    # We need to bound u and v.
    # u^2 * v^3 <= 10^18.
    # v^3 <= 10^18 => v <= 10^6.
    # u^2 <= 10^18 / v^3.

    # Let's generate powerful numbers up to 10^18.
    # To keep the list manageable, we might limit the size.
    # Let's aim for around 100,000 powerful numbers.
    # If v is up to 1000, v^3 = 10^9, u^2 <= 10^9, u <= 31622.
    # Number of pairs (u, v) is roughly 31622 * 1000 = 3 * 10^7.
    # That's a lot, but we can deduplicate.
    # Let's try smaller v first.
    # v up to 100, v^3 = 10^6, u^2 <= 10^12, u <= 10^6.
    # Pairs: 10^6 * 100 = 10^8. Too many.
    
    # Let's just generate them directly using a sieve-like approach or just iterate.
    # Actually, for the purpose of finding N = a + b, we just need a good set.
    # Let's generate powerful numbers up to 10^10 first.
    # v^3 <= 10^10 => v <= 2154.
    # u^2 <= 10^10 / v^3.
    
    # Let's try to generate powerful numbers up to 10^12.
    # v up to 10000 (10^4).
    # u up to sqrt(10^12 / 1) = 10^6.
    # This is getting large.
    
    # Let's stick to the knowledge from the prompt:
    # "Batched np.unique on concatenated sums reduces Python loop overhead by ~500x per batch, enabling larger limit (100M vs 80M) and more powerful numbers"
    # Let's generate powerful numbers up to 10^9.
    # v up to 1000.
    # u up to sqrt(10^9) ~ 31622.
    # Number of generated (u, v) pairs: 31622 * 1000 ~ 3 * 10^7.
    # This is feasible in Python if we vectorize or use C loops, but 30 million is a bit much for pure Python.
    # Let's reduce the limit.
    # Powerful numbers up to 10^7.
    # v up to 215.
    # u up to 3162.
    # Pairs: 3162 * 215 ~ 6.8 * 10^5. This is small.
    # Let's try up to 10^9.
    # We can generate the list in chunks or just accept the time.
    
    # Let's try to generate powerful numbers up to 10^18 efficiently.
    # We will use a set to store them.
    powerful_set = set()
    
    # v_max = 10^6 is too large for a loop.
    # Let's loop v up to 10000.
    for v in range(1, 10001):
        v3 = v ** 3
        if v3 > 10**18:
            break
        # u_max = isqrt(10**18 // v3)
        u_max = isqrt(10**18 // v3)
        # Generate u^2 * v3 for u in 1..u_max
        # To avoid 10^6 iterations in the inner loop for small v, we might need to be smart.
        # But for v=1, u_max = 10^9. That's 1 billion iterations. Too slow.
        
        # We need a different approach.
        # Let's generate powerful numbers by iterating over the square part.
        # n = a^2 * b^3.
        # If we fix b (the cube part), a can be large.
        # Instead, let's just generate a list of powerful numbers up to a smaller limit, say 10^12.
        # v up to 10000.
        # For v=1, u up to 10^6.
        # This is 1 million iterations for v=1.
        # Sum of u_max for v=1 to 10000.
        # u_max = (10^12)^(1/2) / v^(3/2) = 10^6 / v^1.5.
        # Sum_{v=1}^{10000} 10^6 / v^1.5 ~ 10^6 * sum 1/v^1.5.
        # The sum converges to ~2.6.
        # So total iterations ~ 2.6 * 10^6. This is very fast!
        
        # Let's use limit = 10^12.
        limit = 10**12
        for v in range(1, 10001):
            v3 = v ** 3
            if v3 > limit:
                break
            u_max = isqrt(limit // v3)
            # Add u^2 * v3 for u in 1..u_max
            # We can do this in a loop.
            for u in range(1, u_max + 1):
                powerful_set.add(u * u * v3)
                
    powerful_list = sorted(powerful_set)
    powerful_arr = np.array(powerful_list, dtype=np.int64)
    
    # Step 2: Count representations
    # We want to find N that can be written as a + b in many ways, with a, b in powerful_list.
    # We can do this by taking the pairwise sum and counting.
    # But pairwise sum of M elements is M^2/2.
    # If M ~ 10^5, M^2 ~ 10^10. Too big for memory.
    # We need to be smarter.
    
    # Let's use a smaller limit for powerful numbers to keep M small.
    # If limit = 10^9, M is maybe ~ 10^4.
    # M^2 = 10^8. This is manageable with numpy.
    
    # Let's regenerate with limit = 10^9.
    powerful_set = set()
    limit = 10**9
    for v in range(1, 1001): # v^3 <= 10^9 => v <= 1000
        v3 = v ** 3
        if v3 > limit:
            break
        u_max = isqrt(limit // v3)
        for u in range(1, u_max + 1):
            powerful_set.add(u * u * v3)
            
    powerful_list = sorted(powerful_set)
    powerful_arr = np.array(powerful_list, dtype=np.int64)
    M = len(powerful_arr)
    
    # Now, find N = a + b with max representations.
    # We can't do all pairs if M is large.
    # But if M ~ 10^4, M^2 = 10^8.
    # We can batch it.
    
    # Let's estimate M for limit=10^9.
    # Sum_{v=1}^{1000} sqrt(10^9 / v^3) = 10^4.5 * sum 1/sqrt(v^3) = 31622 * sum v^-1.5.
    # sum v^-1.5 ~ 2.6.
    # M ~ 31622 * 2.6 ~ 82000.
    # M^2 ~ 6 * 10^9.
    # This is too big for a single numpy array of int64 (48 GB).
    
    # We need to reduce M.
    # Let's use limit = 10^8.
    powerful_set = set()
    limit = 10**8
    for v in range(1, 500): # v^3 <= 10^8 => v <= 464
        v3 = v ** 3
        if v3 > limit:
            break
        u_max = isqrt(limit // v3)
        for u in range(1, u_max + 1):
            powerful_set.add(u * u * v3)
            
    powerful_list = sorted(powerful_set)
    powerful_arr = np.array(powerful_list, dtype=np.int64)
    M = len(powerful_arr)
    # M ~ sqrt(10^8) * 2.6 ~ 10000 * 2.6 = 26000.
    # M^2 ~ 6 * 10^8.
    # 600 million int64s = 4.8 GB. Still too big.
    
    # Let's reduce further. Limit = 10^7.
    powerful_set = set()
    limit = 10**7
    for v in range(1, 216):
        v3 = v ** 3
        if v3 > limit:
            break
        u_max = isqrt(limit // v3)
        for u in range(1, u_max + 1):
            powerful_set.add(u * u * v3)
            
    powerful_list = sorted(powerful_set)
    powerful_arr = np.array(powerful_list, dtype=np.int64)
    M = len(powerful_arr)
    # M ~ sqrt(10^7) * 2.6 ~ 3162 * 2.6 ~ 8200.
    # M^2 ~ 6 * 10^7.
    # 60 million int64s = 480 MB. This is feasible.
    
    # Let's try limit = 2 * 10^7 to get more.
    powerful_set = set()
    limit = 2 * 10**7
    for v in range(1, 272): # 271^3 = 19.9M
        v3 = v ** 3
        if v3 > limit:
            break
        u_max = isqrt(limit // v3)
        for u in range(1, u_max + 1):
            powerful_set.add(u * u * v3)
            
    powerful_list = sorted(powerful_set)
    powerful_arr = np.array(powerful_list, dtype=np.int64)
    M = len(powerful_arr)
    
    # Now find N with max representations.
    # We will batch the pairwise sums.
    # For each batch of a's, compute a[:, None] + b[None, :]
    # and count unique sums.
    
    best_N = 0
    best_count = 0
    
    # We can optimize by only considering a <= b.
    # But numpy doesn't make this easy.
    # Let's just do all pairs and divide by 2? No, (a,b) and (b,a) are distinct pairs if a!=b, but the representation {a,b} is the same.
    # The task says "distinct pairs a <= b".
    # So we only count each unordered pair once.
    # We can enforce a <= b by iterating.
    
    # Let's use a different approach:
    # For each a in powerful_arr, for each b in powerful_arr where b >= a,
    # add a+b to a counter.
    # This is O(M^2/2).
    # With M ~ 10000, M^2/2 = 5 * 10^7.
    # We can do this in a loop over a, and for each a, vectorize over b.
    
    # Actually, we can just do the full matrix and then use a mask or just accept that (a,b) and (b,a) are counted.
    # If we count all pairs (i, j), then the number of representations for N is count(N) // 2 if N is not a sum of two equal numbers, or (count(N)-1)//2 + 1 if it is.
    # Actually, it's simpler to just count the number of pairs (i, j) with i <= j.
    
    # Let's do it in batches to manage memory.
    # Batch size for a: B_a.
    # For each batch of a's, compute sums with all b's.
    # sums = a_batch[:, None] + b[None, :]
    # This is B_a * M.
    # If B_a = 100, M = 10000, sums is 10^6. This is small.
    
    # We need to count the frequency of each sum.
    # We can use np.unique with return_counts.
    # But we want to accumulate counts over batches.
    # We can't easily accumulate counts in a dict with numpy.
    # We can store the sums in a list and flatten at the end, but that's too much memory.
    # Instead, for each batch, we can find the most frequent sum in that batch, and update the global best.
    # But this is not correct, because a sum might appear in multiple batches.
    
    # Alternative: use a dictionary to store counts.
    # For each batch, get unique sums and counts, and update the dict.
    # This is efficient if the number of unique sums per batch is small.
    
    counts = {}
    batch_size = 100
    for i in range(0, M, batch_size):
        a_batch = powerful_arr[i:i+batch_size]
        # We want pairs (a, b) with a <= b.
        # For a in a_batch, b must be >= a.
        # We can just compute all sums a_batch + b and filter later?
        # No, let's just compute all sums and then filter by a <= b.
        # Actually, it's easier to just iterate a from 0 to M, and b from a to M.
        # This is not easily vectorized in a single numpy op.
        
        # Let's stick to the full matrix approach but in a memory-efficient way.
        # We'll just loop over a.
        pass
        
    # Let's use a pure Python loop for the outer loop, and numpy for the inner.
    # For each a_idx, a = powerful_arr[a_idx]
    # b_candidates = powerful_arr[a_idx:]
    # sums = a + b_candidates
    # unique_sums, counts = np.unique(sums, return_counts=True)
    # for s, c in zip(unique_sums, counts):
    #     counts_dict[s] = counts_dict.get(s, 0) + c
    
    counts_dict = {}
    for a_idx in range(M):
        a = powerful_arr[a_idx]
        b_candidates = powerful_arr[a_idx:]
        sums = a + b_candidates
        unique_sums, local_counts = np.unique(sums, return_counts=True)
        for s, c in zip(unique_sums, local_counts):
            counts_dict[s] = counts_dict.get(s, 0) + int(c)
            
    # Find the best N
    best_N = max(counts_dict, key=counts_dict.get)
    best_count = counts_dict[best_N]
    
    # Extract pairs
    pairs = []
    for a in powerful_list:
        b = best_N - a
        if a <= b and b in powerful_set:
            pairs.append([a, b])
            
    # Verify
    if not pairs:
        return {'N': 0, 'pairs': []}
        
    return {'N': best_N, 'pairs': pairs}
