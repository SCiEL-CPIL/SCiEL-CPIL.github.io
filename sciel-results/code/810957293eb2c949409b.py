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
            
        states[i].append((0, 1, -1, -1))
        p_i = primes[i]
        best_len_at_i = 1
        
        for j in range(i - 1, -1, -1):
            if time.time() > deadline:
                break
            new_gap = p_i - primes[j]
            
            if not states[j]:
                continue
            max_len_j = states[j][-1][1]
            if max_len_j + 1 <= best_len_at_i:
                continue
            
            for state_idx, (g_j, l_j, prev_i, prev_s_idx) in enumerate(states[j]):
                if new_gap < g_j:
                    break
                new_len = l_j + 1
                if new_len <= best_len_at_i:
                    continue
                states[i].append((new_gap, new_len, j, state_idx))
                best_len_at_i = new_len
                
    best_len = 0
    best_i = -1
    best_state_idx = -1
    for i in range(n):
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
