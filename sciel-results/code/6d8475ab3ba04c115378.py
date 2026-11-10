import numpy as np
import time
from collections import defaultdict

def construct():
    N = 10**7
    
    # 1. Precompute sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # 2. Build index: sigma_value -> list of n
    # We only care about sigma values that appear more than once
    sigma_to_ns = defaultdict(list)
    
    # To save memory and time, we only store n if the sigma value is "common"
    # But we don't know which are common until we count.
    # So first pass: count frequencies.
    # Actually, we can just build the dict. N=10^7 is manageable.
    # However, storing lists for every unique sigma might be heavy if there are many unique sigmas.
    # Let's estimate: number of unique sigma values is roughly N / log(N) or similar? No, sigma(n) grows.
    # Actually, just building the dict is O(N).
    
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Build the index
    # To avoid huge memory usage if there are too many unique keys, 
    # we can filter out sigma values that appear only once.
    # But we don't know frequencies yet.
    # Strategy: 
    # 1. Count frequencies of sigma values.
    # 2. Filter indices to only those with freq > 1.
    
    # Counting frequencies
    # Using np.bincount is fast but requires max sigma.
    # Max sigma(n) for n <= 10^7 is around 2.4 * 10^7 (for highly composite numbers).
    # So we can use a dictionary or array.
    # Dictionary is safer.
    
    freq = defaultdict(int)
    for n in range(1, N + 1):
        s = sigma[n]
        freq[s] += 1
        
    if time.time() > deadline:
        # Fallback
        return [11, 6]
        
    # Now, only keep sigma values with freq > 1
    # And build the mapping
    sigma_to_ns = defaultdict(list)
    for n in range(1, N + 1):
        s = sigma[n]
        if freq[s] > 1:
            sigma_to_ns[s].append(n)
            
    if time.time() > deadline:
        return [11, 6]
        
    # 3. Search for best pair
    best_a = 1
    best_b = 1
    best_err = float('inf')
    
    # We want to minimize |a/b - 3/2|.
    # Let a = 3b/2 + delta.
    # |delta| / b is the error.
    # delta = a - 1.5b.
    # 2a = 3b + 2delta.
    # Let k = 2delta. Then 2a - 3b = k.
    # k must be an integer.
    # Error = |k| / (2b).
    
    # We iterate k from 1 upwards.
    # For a fixed k, we want to maximize b to minimize error k/(2b).
    # Constraints:
    # 1. a, b <= N
    # 2. sigma(a) == sigma(b)
    # 3. 2a - 3b = k  => a = (3b + k) / 2
    #    So 3b + k must be even.
    #    Since 3 is odd, b and k must have same parity.
    
    # We will iterate k = 1, 2, 3, ...
    # For each k, we check possible b values.
    # To be efficient, we can iterate b in a smart way?
    # Or iterate over the groups in sigma_to_ns?
    
    # Alternative approach:
    # For each group of numbers with the same sigma value S, let the set be G_S.
    # We want to find a, b in G_S such that |a/b - 1.5| is minimized.
    # This is equivalent to minimizing |2a - 3b| / (2b).
    
    # If we iterate over all groups, and for each group check all pairs, it's O(sum |G_S|^2).
    # This could be too slow if groups are large.
    
    # Let's stick to the k-iteration but optimize the check.
    # For a given k, a = (3b+k)/2.
    # We need b such that b and a are in the same sigma group.
    # This means sigma(b) == sigma((3b+k)/2).
    
    # We can iterate b from N down to 1?
    # No, we want the largest b for a given k to minimize error.
    # So for a given k, we can scan b from N down to 1, step 2 (to match parity).
    # Check if sigma(b) == sigma((3b+k)/2).
    # If yes, we found the best b for this k.
    # Then we can stop checking this k?
    # Yes, because for fixed k, error decreases as b increases.
    # So the first match (largest b) is the best for this k.
    
    # But we need to check if a <= N.
    # a = (3b+k)/2 <= N => 3b+k <= 2N => b <= (2N-k)/3.
    
    # So for each k, we scan b from min(N, floor((2N-k)/3)) down to 1, step 2.
    
    # How many k do we need to check?
    # The error is k/(2b).
    # If we have a current best_err, we stop if k/(2*1) > best_err? No.
    # The minimum possible error for a given k is k/(2*N).
    # So if k/(2*N) >= best_err, we can stop.
    # k >= 2*N*best_err.
    
    # Let's try this.
    
    max_k = 10000  # Upper bound just in case
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
            
        # Lower bound check
        if best_err < k / (2.0 * N):
            break
            
        parity_k = k % 2
        
        # Max b such that a <= N
        b_max = int((2 * N - k) / 3)
        if b_max > N:
            b_max = N
            
        # Adjust b_max to match parity of k
        if b_max % 2 != parity_k:
            b_max -= 1
            
        # Scan b downwards
        found = False
        for b in range(b_max, 0, -2):
            a = (3 * b + k) // 2
            # a is integer by construction of parity
            if a > N:
                continue
            if sigma[b] == sigma[a]:
                # Found a pair
                err = k / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                found = True
                break # Found largest b for this k, stop scanning b for this k
        
        # Optional: If we found a very good pair, we might break early?
        # The lower bound check handles this.
        
    # Fallback
    if best_a == 1 and best_b == 1:
        best_a = 11
        best_b = 6
        best_err = abs(11/6 - 1.5)
        
    return [int(best_a), int(best_b)]
