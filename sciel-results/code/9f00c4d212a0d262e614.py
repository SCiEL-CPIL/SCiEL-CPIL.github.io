"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import Counter, defaultdict

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000
    WINDOW = 24
    MAX_VAL = N + WINDOW
    
    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Step 1: Check for distinct count = 1
    # This requires s[n+i] = 1 for all i in 1..24
    # i.e., n+i is coprime to P = prod(PRIMES)
    
    is_coprime = np.ones(MAX_VAL + 1, dtype=bool)
    indices = np.arange(MAX_VAL + 1)
    for p in PRIMES:
        is_coprime[indices % p == 0] = False
        
    # Sliding window sum of is_coprime
    prefix_sum = np.cumsum(is_coprime)
    
    left = np.arange(1000, MAX_VAL - 23) # i-1
    right = np.arange(1024, MAX_VAL + 1) # i+23
    
    window_sums = prefix_sum[right] - prefix_sum[left]
    
    matches = np.where(window_sums == 24)[0]
    if len(matches) > 0:
        k = matches[0]
        i = 1001 + k
        n = i - 1
        return n

    # Step 2: Compute s[m] = product of p^v_p(m) for p in PRIMES
    # More efficient: compute the "smooth part" using a modified sieve
    s = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in PRIMES:
        # For each multiple of p, multiply by the highest power of p dividing it
        power = p
        while power <= MAX_VAL:
            next_power = power * p
            # Indices divisible by power but not by next_power have exact p-adic valuation corresponding to power
            mask = (indices % power == 0)
            if next_power <= MAX_VAL:
                mask &= (indices % next_power != 0)
            s[mask] *= power
            power = next_power

    # Assign integer IDs to distinct s values for faster distinct counting
    # Map each unique s value to a small integer
    unique_s = np.unique(s[1:])  # s[0] is unused
    # Create a mapping from s value to ID
    # Since s values can be large, use a dict
    s_to_id = {}
    id_to_s = {}
    next_id = 0
    for val in unique_s:
        s_to_id[int(val)] = next_id
        id_to_s[next_id] = int(val)
        next_id += 1
    
    # Convert s array to ID array using vectorized approach
    s_ids = np.zeros(MAX_VAL + 1, dtype=np.int32)
    s_ids[1:] = np.vectorize(lambda x: s_to_id[int(x)])(s[1:])
    
    # Now do sliding window with integer IDs - much faster
    best_n = 1000
    best_distinct = WINDOW + 1
    
    # Initialize window for n=1000: indices 1001 to 1024
    freq = defaultdict(int)
    for i in range(1001, 1025):
        freq[s_ids[i]] += 1
        
    distinct = len(freq)
    best_distinct = distinct
    best_n = 1000
    
    if best_distinct == 2:
        return best_n

    # Slide window: n goes from 1001 to N
    # Remove s_ids[n], add s_ids[n+24]
    for n in range(1001, N + 1):
        if time.time() > deadline:
            break
            
        # Remove s_ids[n]
        v_out = s_ids[n]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            
        # Add s_ids[n+24]
        v_in = s_ids[n + 24]
        freq[v_in] += 1
        
        distinct = len(freq)
        
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct == 2:
                return best_n

    return best_n

# EVOLVE-BLOCK-END
