def construct():
    from math import isqrt
    import time
    
    deadline = time.time() + 55
    
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, N+1) if is_prime[i]]
    n = len(primes)
    
    states = [[] for _ in range(n)]
    
    for i in range(n):
        if time.time() > deadline:
            break
            
        # Start state
        states[i].append((0, 1, -1, -1))
        
        p_i = primes[i]
        # Iterate j in reverse to ensure new_gap is increasing
        for j in range(i - 1, -1, -1):
            new_gap = p_i - primes[j]
            
            # Iterate states of j
            # states[j] is sorted by gap increasing
            # If new_gap < g_j, break
            for state_idx, (g_j, l_j, prev_i, prev_s_idx) in enumerate(states[j]):
                if new_gap < g_j:
                    break
                new_len = l_j + 1
                
                # Check dominance
                # states[i] is sorted by gap increasing, len increasing
                # Last element has max gap and max len
                if states[i][-1][1] >= new_len:
                    # Dominated by the state with max len (which has gap <= new_gap)
                    # Actually, we need to be careful.
                    # The last state has gap g_last <= new_gap (since we add in increasing gap order).
                    # And len_last is the max len so far.
                    # If len_last >= new_len, then (g_last, len_last) dominates (new_gap, new_len)
                    # because g_last <= new_gap and len_last >= new_len.
                    continue
                else:
                    states[i].append((new_gap, new_len, j, state_idx))
                    
    # Find best
    best_len = 0
    best_i = -1
    best_state_idx = -1
    for i in range(n):
        # The last state in states[i] has the max length for that i
        # But we need global max
        if states[i]:
            l = states[i][-1][1]
            if l > best_len:
                best_len = l
                best_i = i
                best_state_idx = len(states[i]) - 1
                
    if best_i == -1:
        return [2]
        
    result = []
    cur_i = best_i
    cur_state_idx = best_state_idx
    while cur_i != -1:
        result.append(primes[cur_i])
        _, _, prev_i, prev_s_idx = states[cur_i][cur_state_idx]
        cur_i = prev_i
        cur_state_idx = prev_s_idx
        
    result.reverse()
    return result
