def construct():
    from math import isqrt
    import time
    
    deadline = time.time() + 55  # safety margin before 90s kill
    
    # Sieve of Eratosthenes for primes <= 10000
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, N+1) if is_prime[i]]
    n = len(primes)
    
    # DP with proper path tracking
    # states[i] = list of (last_gap, length, prev_prime_idx, prev_state_idx)
    # Non-dominated: sorted by last_gap increasing, length strictly increasing
    
    states = [[] for _ in range(n)]
    
    for i in range(n):
        if time.time() > deadline:
            break
        # Start a new sequence at prime i
        states[i].append((0, 1, -1, -1))
        
        # Extend sequences ending at j < i
        for j in range(i):
            new_gap = primes[i] - primes[j]
            # For each state in states[j], check if new_gap >= last_gap
            for state_idx, (g_j, l_j, _, _) in enumerate(states[j]):
                if new_gap >= g_j:
                    new_len = l_j + 1
                    # Check if (new_gap, new_len) is dominated by existing states in states[i]
                    dominated = False
                    for (g_e, l_e, _, _) in states[i]:
                        if g_e <= new_gap and l_e >= new_len:
                            dominated = True
                            break
                    if not dominated:
                        states[i].append((new_gap, new_len, j, state_idx))
        
        # Prune states[i]: keep only non-dominated
        if len(states[i]) > 1:
            states[i].sort(key=lambda x: (x[0], -x[1]))
            pruned = []
            max_len_so_far = 0
            for state in states[i]:
                if state[1] > max_len_so_far:
                    pruned.append(state)
                    max_len_so_far = state[1]
            states[i] = pruned
    
    # Find the state with maximum length
    best_len = 0
    best_i = -1
    best_state_idx = -1
    for i in range(n):
        for state_idx, (g, l, _, _) in enumerate(states[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_state_idx = state_idx
    
    if best_i == -1:
        return [2]
    
    # Reconstruct the path
    result = []
    cur_i = best_i
    cur_state_idx = best_state_idx
    while cur_i != -1:
        result.append(primes[cur_i])
        _, _, prev_i, prev_state_idx = states[cur_i][cur_state_idx]
        cur_i = prev_i
        cur_state_idx = prev_state_idx
    
    result.reverse()
    return result
