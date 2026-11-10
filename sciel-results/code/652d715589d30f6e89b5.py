"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math

    # 1. Generate all primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # 2. Dynamic Programming with Pareto-optimal states
    # states[i] is a list of (last_gap, length, prev_index)
    # The list is sorted by last_gap ascending.
    # For non-dominated states, as last_gap increases, length must strictly increase.
    states = [[] for _ in range(n)]
    
    # Initialize states
    for i in range(n):
        states[i] = [(0, 1, -1)]
        
    for i in range(n):
        pi = primes[i]
        current_states = states[i]
        
        # Collect all potential new states from predecessors
        new_candidates = []
        for j in range(i):
            pj = primes[j]
            d = pi - pj
            
            for gap_j, len_j, prev_j in states[j]:
                if gap_j <= d:
                    new_len = len_j + 1
                    new_gap = d
                    new_candidates.append((new_gap, new_len, j))
        
        if not new_candidates:
            continue
            
        # Merge current_states and new_candidates
        # Combine and sort by gap
        combined = current_states + new_candidates
        combined.sort(key=lambda x: x[0])
        
        # Prune dominated states
        pruned = []
        max_len_so_far = -1
        last_gap = -1
        last_len = -1
        
        for gap, length, prev in combined:
            if gap == last_gap:
                if length > last_len:
                    if pruned and pruned[-1][0] == gap:
                        pruned[-1] = (gap, length, prev)
                        last_len = length
                        if length > max_len_so_far:
                            max_len_so_far = length
            else:
                if length > max_len_so_far:
                    pruned.append((gap, length, prev))
                    last_gap = gap
                    last_len = length
                    max_len_so_far = length
        
        states[i] = pruned

    # 3. Find the end of the longest sequence
    max_len = 0
    end_idx = -1
    
    for i in range(n):
        for gap, length, prev in states[i]:
            if length > max_len:
                max_len = length
                end_idx = i
                
    if end_idx == -1:
        return []

    # 4. Reconstruct the sequence
    seq = []
    cur_idx = end_idx
    cur_len = max_len
    
    while cur_idx != -1:
        seq.append(primes[cur_idx])
        
        # Find the state in states[cur_idx] with length == cur_len
        # There should be exactly one such state in the Pareto set
        prev_idx = -1
        for gap, length, prev in states[cur_idx]:
            if length == cur_len:
                prev_idx = prev
                break
        
        cur_idx = prev_idx
        cur_len -= 1
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
