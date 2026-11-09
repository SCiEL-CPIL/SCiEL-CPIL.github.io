"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from math import isqrt

    # Generate all primes <= 10000 using a simple sieve
    limit = 10000
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    
    primes = np.where(sieve)[0]
    n = len(primes)
    
    if n == 0:
        return []
    
    # We want the longest subsequence of primes p_1 < p_2 < ... < p_k
    # such that gaps g_i = p_{i+1} - p_i are nondecreasing: g_1 <= g_2 <= ... <= g_{k-1}
    
    # DP approach:
    # Let dp[i] = length of the longest valid sequence ending at primes[i]
    # Let last_gap[i] = the gap of the last step in the optimal sequence ending at primes[i]
    # But we need to be careful: for a given ending prime, there might be multiple
    # optimal-length sequences with different last gaps. We want to maximize length,
    # and among those, we want the smallest last gap to allow more extensions.
    
    # Actually, let's think more carefully. We want to maximize length.
    # dp[i] = maximum length of a valid nondecreasing-gap prime subsequence ending at primes[i]
    # To compute dp[i], we look at all j < i where primes[i] - primes[j] >= last_gap[j]
    # and take max(dp[j] + 1).
    # But last_gap[j] depends on which predecessor was chosen for j.
    
    # Better approach: 
    # dp[i] = (length, last_gap) but we need to track the best (max length, then min last_gap)
    # for each ending prime.
    
    # Let's define:
    # best_len[i] = maximum length of valid sequence ending at primes[i]
    # best_gap[i] = minimum last gap among all sequences of length best_len[i] ending at primes[i]
    
    # For i from 0 to n-1:
    #   best_len[i] = 1, best_gap[i] = infinity (single element, no gap)
    #   For j from 0 to i-1:
    #     gap = primes[i] - primes[j]
    #     If best_len[j] == 1 or gap >= best_gap[j]:
    #       new_len = best_len[j] + 1
    #       if new_len > best_len[i] or (new_len == best_len[i] and gap < best_gap[i]):
    #         best_len[i] = new_len
    #         best_gap[i] = gap
    
    # This is O(n^2) which for ~1229 primes is about 1.5 million operations, very fast.
    
    best_len = np.ones(n, dtype=int)
    best_gap = np.full(n, np.inf, dtype=float)
    
    # For single-element sequences, there's no gap, so we treat it specially
    # A sequence of length 1 has no gap constraint to satisfy for extension
    
    for i in range(1, n):
        pi = primes[i]
        max_len = 1
        min_gap = np.inf
        
        for j in range(i):
            gap = pi - primes[j]
            # Check if we can extend the sequence ending at j
            if best_len[j] == 1:
                # Single element, always valid to extend
                new_len = best_len[j] + 1
                if new_len > max_len or (new_len == max_len and gap < min_gap):
                    max_len = new_len
                    min_gap = gap
            else:
                # Need gap >= best_gap[j]
                if gap >= best_gap[j]:
                    new_len = best_len[j] + 1
                    if new_len > max_len or (new_len == max_len and gap < min_gap):
                        max_len = new_len
                        min_gap = gap
        
        best_len[i] = max_len
        best_gap[i] = min_gap
    
    # Find the maximum length
    max_length = int(np.max(best_len))
    
    # Reconstruct the sequence
    # We need to find a sequence of length max_length
    # Start from an index i where best_len[i] == max_length
    # Then backtrack
    
    # Pick the last element: among all i with best_len[i] == max_length, 
    # pick the one with smallest best_gap to be safe, or just pick any
    end_idx = np.argmax(best_len)
    # Actually, we want the last element of our sequence. Let's find all candidates
    candidates = np.where(best_len == max_length)[0]
    # Pick the one with smallest last gap for flexibility, though it doesn't matter for reconstruction
    end_idx = candidates[np.argmin(best_gap[candidates])]
    
    # Backtrack
    sequence = [primes[end_idx]]
    current_idx = end_idx
    current_gap = best_gap[end_idx]
    current_len = best_len[end_idx]
    
    while current_len > 1:
        # Find j < current_idx such that:
        #   primes[current_idx] - primes[j] == current_gap (or the gap that led to current state)
        #   best_len[j] == current_len - 1
        #   And the gap from j to current_idx is >= best_gap[j]
        # Actually, we need to find the predecessor that was used
        # The gap into current_idx is current_gap
        # We need j such that primes[current_idx] - primes[j] == current_gap? No, 
        # the gap is primes[current_idx] - primes[j], and this must equal current_gap only if
        # current_gap is the gap of the last step.
        
        # Wait, best_gap[current_idx] is the gap of the last step in the optimal sequence ending at current_idx.
        # So we need j such that primes[current_idx] - primes[j] == best_gap[current_idx]
        # AND best_len[j] == best_len[current_idx] - 1
        # AND (best_len[j] == 1 OR best_gap[current_idx] >= best_gap[j])
        
        target_gap = best_gap[current_idx]
        found = False
        for j in range(current_idx):
            if primes[current_idx] - primes[j] == target_gap:
                if best_len[j] == current_len - 1:
                    if best_len[j] == 1 or target_gap >= best_gap[j]:
                        sequence.append(primes[j])
                        current_idx = j
                        current_gap = best_gap[j]
                        current_len = best_len[j]
                        found = True
                        break
        if not found:
            # This shouldn't happen, but break to avoid infinite loop
            break
    
    sequence.reverse()
    return list(sequence)
# EVOLVE-BLOCK-END
