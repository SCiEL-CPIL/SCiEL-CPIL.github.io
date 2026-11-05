import time

def _sieve(n):
    is_p = bytearray(b'\x01') * (n + 1)
    is_p[0:2] = b'\x00\x00'
    for p in range(2, int(n ** 0.5) + 1):
        if is_p[p]:
            start = p * p
            is_p[start:n + 1:p] = b'\x00' * (((n - start) // p) + 1)
    return [i for i in range(2, n + 1) if is_p[i]]

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    
    primes = _sieve(10000)
    n = len(primes)
    
    if n <= 1:
        return primes

    # states[j] is a list of (length, gap, prev_idx, prev_state_idx)
    # We store these to reconstruct.
    states = [[] for _ in range(n)]
    
    # Global best
    best_len = 0
    best_gap = 0
    best_j = -1
    best_state_idx = -1

    for j in range(n):
        pj = primes[j]
        # Initialize with the sequence of just [pj]
        # Length 1, Gap 0
        curr_states = [(1, 0, -1, -1)]
        
        for i in range(j):
            pi = primes[i]
            # For each state in i, see if it can extend to j
            for k in range(len(states[i])):
                len_i, gap_i, prev_i, prev_k = states[i][k]
                gap = pj - pi
                if gap >= gap_i:
                    # New candidate
                    new_len = len_i + 1
                    new_gap = gap
                    curr_states.append((new_len, new_gap, i, k))
        
        # Prune curr_states to Pareto optimal
        # Sort by length descending, then gap ascending
        curr_states.sort(key=lambda x: (-x[0], x[1]))
        
        pruned = []
        min_gap_so_far = float('inf')
        for state in curr_states:
            l, g, pi, ki = state
            if g < min_gap_so_far:
                pruned.append(state)
                min_gap_so_far = g
        
        states[j] = pruned
        
        # Update global best
        # The best state for j is the first one in pruned (longest, then smallest gap)
        if pruned:
            l, g, pi, ki = pruned[0]
            if l > best_len or (l == best_len and g < best_gap):
                best_len = l
                best_gap = g
                best_j = j
                best_state_idx = 0 # Index in states[j]

        if time.time() - t0 > DEADLINE:
            break

    # Reconstruct
    seq = []
    cur_j = best_j
    cur_k = best_state_idx
    while cur_j != -1:
        l, g, prev_j, prev_k = states[cur_j][cur_k]
        seq.append(primes[cur_j])
        cur_j = prev_j
        cur_k = prev_k
    
    seq.reverse()
    return seq
