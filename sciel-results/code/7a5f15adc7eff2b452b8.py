import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sieve for primes up to 10000
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.array([i for i, v in enumerate(is_prime) if v])
    N = len(primes)
    
    if N == 0:
        return []
        
    # Max possible length is N, but practically small.
    # We can set a reasonable upper bound. 
    # For primes <= 10000, the longest non-decreasing gap sequence is likely < 30.
    # Let's set L_max = 50 to be safe.
    L_max = 50
    
    # dp[i, l] = min last gap for sequence of length l+1 ending at i
    # l=0 corresponds to length 1.
    # parent[i, l] = j (previous index) for the sequence that achieved dp[i, l]
    INF = 10**9
    
    dp = np.full((N, L_max), INF, dtype=np.int64)
    parent = np.full((N, L_max), -1, dtype=np.int64)
    
    # Initialize length 1 (l=0)
    dp[:, 0] = 0
    # parent for l=0 is -1
    
    best_len = 1
    best_index = 0
    
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        # gaps from all previous primes j < i to prime i
        gaps = primes[i] - primes[:i]
        
        # We iterate through lengths l
        # dp[i, l+1] depends on dp[j, l] for j < i
        # We can update all l in a loop, but vectorize over j
        
        # For each length l, we find the best j
        # We process l from 0 to L_max-2 (to update l+1)
        for l in range(L_max - 1):
            if time.time() > deadline:
                break
                
            # dp[:i, l] contains the min last gaps for sequences of length l+1 ending at j < i
            # gaps contains the gap from j to i
            # We need dp[j, l] <= gaps[j]
            
            prev_dp = dp[:i, l]
            
            # Mask of valid predecessors
            valid_mask = prev_dp <= gaps
            
            if np.any(valid_mask):
                # Get the indices of valid predecessors
                valid_indices = np.where(valid_mask)[0]
                
                # Get the gaps for these valid predecessors
                valid_gaps = gaps[valid_indices]
                
                # Find the predecessor with the minimum gap
                # This minimizes the last gap for the new sequence, which is good for future extensions
                min_gap_idx = np.argmin(valid_gaps)
                min_gap_val = valid_gaps[min_gap_idx]
                best_j = valid_indices[min_gap_idx]
                
                # Update dp[i, l+1] if this is better (smaller gap)
                if min_gap_val < dp[i, l + 1]:
                    dp[i, l + 1] = min_gap_val
                    parent[i, l + 1] = best_j
                    
                    # Update best length
                    new_len = l + 2 # length is l+1 for prev, so l+2 for current
                    if new_len > best_len:
                        best_len = new_len
                        best_index = i

    # Reconstruct
    if best_len == 1:
        return [int(primes[best_index])]
        
    path = []
    curr_idx = best_index
    curr_len = best_len - 1 # index in dp array (0-based length)
    
    while curr_len >= 0:
        path.append(int(primes[curr_idx]))
        if curr_len == 0:
            break
        prev_idx = parent[curr_idx, curr_len]
        curr_idx = prev_idx
        curr_len -= 1
        
    path.reverse()
    return path
