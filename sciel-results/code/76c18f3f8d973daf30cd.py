import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    LIMIT = 10000
    DEADLINE = start_time + 85.0

    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # dp[i] is a list of tuples: (gap, length, parent_idx, parent_state_idx)
    # parent_idx is the index in primes list (j)
    # parent_state_idx is the index in dp[parent_idx]
    dp = [[] for _ in range(n)]
    
    # Initialize first prime
    # Gap 0, Length 1, No parent
    dp[0] = [(0, 1, -1, -1)]
    
    best_len = 1
    best_i = 0
    best_state_idx = 0
    
    for i in range(1, n):
        if time.time() > DEADLINE:
            break
            
        p_i = primes[i]
        cands = []
        
        for j in range(i):
            gap_new = p_i - primes[j]
            
            # dp[j] is sorted by gap
            # We want the state in dp[j] with max length such that gap <= gap_new
            # Since dp[j] is Pareto optimal, lengths are non-decreasing with gaps.
            # So we find the rightmost state with gap <= gap_new.
            
            # Extract gaps for binary search
            # Optimization: pre-extract gaps? Or just bisect on list of tuples?
            # bisect works on lists. We can create a list of gaps.
            # Or we can just loop if dp[j] is small.
            # Let's assume dp[j] is small (< 100).
            # To be safe and fast, we can maintain a separate list of gaps for each dp[i]?
            # Or just use bisect on a list of gaps.
            
            # For simplicity and speed, let's just iterate if len(dp[j]) is small, 
            # or use bisect if we store gaps separately.
            # Let's store dp[i] as list of (gap, length, p_idx, p_state_idx)
            # And maintain a parallel list of gaps for bisect?
            # Or just bisect on the list of tuples using a key? 
            # Python's bisect doesn't take a key in older versions, 
            # but we can use bisect_right on a list of just gaps.
            
            # Let's create gaps_j = [s[0] for s in dp[j]]
            # This is O(len(dp[j])).
            # Total O(N^2 * K).
            
            gaps_j = [s[0] for s in dp[j]]
            
            # Find index
            # bisect_right returns insertion point.
            # We want the last index where gap <= gap_new.
            # bisect_right(gaps_j, gap_new) gives index to insert to keep order.
            # So the index we want is pos - 1.
            pos = bisect.bisect_right(gaps_j, gap_new)
            
            if pos > 0:
                best_state_in_j = pos - 1
                # Check if this state is valid (it is, by construction of bisect)
                # Get length
                l_prev = dp[j][best_state_in_j][1]
                l_new = l_prev + 1
                # Candidate state
                cands.append((gap_new, l_new, j, best_state_in_j))
        
        if not cands:
            # Should not happen as j=0 is always valid if gap > 0
            # Unless primes[i] == primes[0] which is false for i>0
            # But if no valid parent, we can start a new sequence of length 1?
            # The problem asks for subsequence. A single element is a valid subsequence.
            # But we want max length.
            # If we can't extend any, we can just have [p_i] with length 1.
            # However, any prime can follow 2 if gap is large enough?
            # Yes.
            # So cands should never be empty for i>0.
            cands.append((0, 1, -1, -1))

        # Filter cands to Pareto optimal
        # Sort by gap
        cands.sort(key=lambda x: x[0])
        
        filtered = []
        max_len_so_far = 0
        for state in cands:
            gap, length, p_idx, p_state_idx = state
            if length > max_len_so_far:
                filtered.append(state)
                max_len_so_far = length
        
        dp[i] = filtered
        
        # Update global best
        if max_len_so_far > best_len:
            best_len = max_len_so_far
            best_i = i
            best_state_idx = len(filtered) - 1 # Index of the last (max length) state
            
    # Reconstruct
    if best_len == 1:
        return [primes[best_i]]
        
    seq = []
    i = best_i
    state_idx = best_state_idx
    
    while i != -1:
        seq.append(primes[i])
        if i == 0:
            break
        # Get parent info
        state = dp[i][state_idx]
        p_idx = state[2]
        p_state_idx = state[3]
        i = p_idx
        state_idx = p_state_idx
        
    seq.reverse()
    return seq
