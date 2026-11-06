import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N_LIMIT = 10000
    
    # 1. Sieve of Eratosthenes
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N_LIMIT) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    if n < 2:
        return primes
    
    # dp[i] is a list of tuples (length, last_gap, prev_index)
    # We maintain that for any fixed length, only the state with the smallest last_gap is kept.
    # Thus, in dp[i], all lengths are distinct.
    dp = [[] for _ in range(n)]
    dp[0] = [(1, 0, -1)]
    
    best_len = 1
    best_idx = 0
    best_state = (1, 0, -1)
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        p_i = primes[i]
        # Start with the state representing the sequence [p_i]
        # Length 1, gap 0, no prev
        new_states = [(1, 0, -1)]
        
        for j in range(i):
            gap = p_i - primes[j]
            # For each state in dp[j], check if it can extend to i
            for (l, g, pi) in dp[j]:
                if gap >= g:
                    # New state: length l+1, last gap is gap, prev is j
                    new_states.append((l + 1, gap, j))
        
        # Filter new_states to keep only non-dominated states
        # Sort by length descending, then gap ascending
        # If two states have same length, the one with smaller gap dominates.
        # If state A has length > B and gap <= B, A dominates B.
        
        # Efficient filtering:
        # Sort by length descending.
        new_states.sort(key=lambda x: (-x[0], x[1]))
        
        filtered = []
        # We iterate through sorted states.
        # Since sorted by length desc, the first state of a given length is the best (smallest gap) for that length.
        # We need to check if it is dominated by any previously added state (which has >= length).
        # Actually, if we iterate length desc, any previously added state has length >= current length.
        # If a previous state has length >= current length and gap <= current gap, current is dominated.
        
        for s in new_states:
            l, g, pi = s
            dominated = False
            for fs in filtered:
                fl, fg, _ = fs
                if fl >= l and fg <= g:
                    dominated = True
                    break
            if not dominated:
                filtered.append(s)
                
        dp[i] = filtered
        
        if filtered:
            # The first state in filtered is the one with max length (and min gap for that length)
            # Because we sorted by -length, the first element has the max length.
            best_local = filtered[0]
            if best_local[0] > best_len:
                best_len = best_local[0]
                best_idx = i
                best_state = best_local

    # Reconstruct sequence
    if best_len == 1:
        return [primes[best_idx]]
    
    path_indices = []
    curr_idx = best_idx
    curr_len = best_state[0]
    curr_gap = best_state[1]
    curr_prev = best_state[2]
    
    while curr_idx != -1:
        path_indices.append(curr_idx)
        if curr_len == 1:
            break
        
        # Find the state in dp[curr_prev] that has length curr_len - 1
        # And whose gap is <= curr_gap
        # Note: curr_prev is the index of the previous prime.
        # The state at curr_prev that led to this one must have length curr_len - 1.
        # As established, there is at most one state of a given length in dp[curr_prev].
        
        target_len = curr_len - 1
        found_state = None
        for (l, g, pi) in dp[curr_prev]:
            if l == target_len:
                # Check if this state is valid predecessor
                # The gap g must be <= curr_gap
                if g <= curr_gap:
                    found_state = (l, g, pi)
                    break
        
        if found_state is None:
            # This should not happen if DP is correct
            # But just in case, break to avoid infinite loop or error
            break
            
        curr_idx = curr_prev
        curr_len = target_len
        curr_gap = found_state[1]
        curr_prev = found_state[2]
        
    path_indices.reverse()
    return [primes[i] for i in path_indices]
