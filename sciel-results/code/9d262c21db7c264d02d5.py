import bisect
import time

def construct():
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes

    # dp[i] will be a list of tuples: (gap, length, prev_prime_idx, prev_idx_in_dp)
    # prev_prime_idx is -1 if this is the start of the sequence (length 1)
    # prev_idx_in_dp is the index in dp[prev_prime_idx] of the state used
    dp = [None] * N
    # sorted_dp[i] is dp[i] sorted by gap
    # prefix_info[i] is a list of (gap, max_len, idx_in_dp) for binary search
    # We store prefix_info to allow O(log N) queries
    sorted_dp = [None] * N
    prefix_info = [None] * N

    for i in range(N):
        # Initialize with the state of length 1 (gap 0, prev -1)
        curr_dp = [(0, 1, -1, -1)]
        
        for j in range(i):
            gap = primes[i] - primes[j]
            
            # Query sorted_dp[j] for max length with key <= gap
            # prefix_info[j] is sorted by gap
            # We need the rightmost element with gap <= current gap
            # Let's use bisect_right on the list of gaps
            # prefix_info[j] is list of (gap, max_len, idx)
            # We can extract gaps list for bisect, or use a custom key
            # To be efficient, let's keep gaps separate or just use bisect on the list of tuples?
            # bisect works on lists of comparable items. Tuples are compared lexicographically.
            # We want to find the largest gap <= G.
            # bisect_right(list_of_gaps, G) gives the index to insert G to keep order.
            # The element at index-1 is the largest <= G.
            
            # To avoid extracting gaps list every time (or storing it),
            # we can store `gaps_list[j]` separately.
            
            # Let's optimize: store `gaps_list[j]` and `prefix_info[j]` (max_len, idx)
            # Actually, prefix_info[j] can just be `max_lens[j]` and `best_idxs[j]`
            # aligned with `gaps_list[j]`.
            
            # Let's restructure storage:
            # gaps_list[j]: sorted list of gaps in dp[j]
            # max_lens[j]: prefix max of lengths for those gaps
            # best_idxs[j]: index in dp[j] that achieved that max length
            
            # We will build these after curr_dp is complete.
            # For now, we need to query j.
            
            # Since we are in loop for i, and j < i, sorted_dp[j] is already computed.
            
            # We need `gaps_list[j]`
            # Let's add `gaps_list` array
            pass

    # Let's rewrite with separate arrays for efficiency
    gaps_list = [None] * N
    max_lens = [None] * N
    best_idxs = [None] * N

    for i in range(N):
        curr_dp = [(0, 1, -1, -1)]
        
        for j in range(i):
            gap = primes[i] - primes[j]
            
            # Binary search in gaps_list[j]
            # Find rightmost index k such that gaps_list[j][k] <= gap
            # bisect_right returns insertion point.
            k = bisect.bisect_right(gaps_list[j], gap)
            
            if k > 0:
                # The best state is at index k-1 in the sorted list
                # But we need the max length up to k-1.
                # max_lens[j][k-1] is the max length for gaps <= gaps_list[j][k-1]
                # Which is <= gap.
                best_len = max_lens[j][k-1]
                best_idx_in_dp_j = best_idxs[j][k-1]
                
                new_len = best_len + 1
                curr_dp.append((gap, new_len, j, best_idx_in_dp_j))
            else:
                # No valid predecessor?
                # But gap 0 is always in dp[j] (length 1).
                # And gap = primes[i] - primes[j] >= 2 (for i>j, distinct primes)
                # So gaps_list[j] contains 0.
                # 0 <= gap is always true.
                # So k should be at least 1.
                # This case shouldn't happen unless primes are not sorted or something.
                pass

        dp[i] = curr_dp
        
        # Prepare structures for future queries
        # Sort curr_dp by gap
        curr_dp_sorted = sorted(curr_dp, key=lambda x: x[0])
        gaps_list[i] = [x[0] for x in curr_dp_sorted]
        
        max_len_so_far = 0
        max_lens_list = []
        best_idxs_list = []
        
        for idx, state in enumerate(curr_dp_sorted):
            # state is (gap, length, prev_prime_idx, prev_idx_in_dp)
            if state[1] > max_len_so_far:
                max_len_so_far = state[1]
            max_lens_list.append(max_len_so_far)
            best_idxs_list.append(idx)
            
        max_lens[i] = max_lens_list
        best_idxs[i] = best_idxs_list

    # Find the best state
    max_len_global = 0
    best_i = -1
    best_idx_in_dp_i = -1
    
    for i in range(N):
        # The max length for i is max_lens[i][-1]
        # But we need the specific state that achieved it.
        # Actually, max_lens[i][-1] is the max length of ANY state in dp[i].
        # The state that achieved it is dp[i][best_idxs[i][-1]]
        if max_lens[i][-1] > max_len_global:
            max_len_global = max_lens[i][-1]
            best_i = i
            best_idx_in_dp_i = best_idxs[i][-1]

    # Reconstruct
    path = []
    cur_i = best_i
    cur_idx = best_idx_in_dp_i
    
    while cur_i != -1:
        state = dp[cur_i][cur_idx]
        path.append(primes[cur_i])
        prev_i = state[2]
        prev_idx = state[3]
        cur_i = prev_i
        cur_idx = prev_idx
        
    path.reverse()
    return path
