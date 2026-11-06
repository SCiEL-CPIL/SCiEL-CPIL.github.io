<<<<<<< SEARCH
def _compute_threshold(trans):
    """
    Compute exact shortest reset word length via BFS on subset automaton.
    trans: list of two lists, trans[letter][state] = dest state.
    Returns (threshold_length, is_synchronizing).
    """
    full = (1 << N) - 1  # bitmask for all states
    # Precompute subset transitions for speed
    # For each letter and each subset (bitmask), compute image subset
    # But 2^12 * 2 = 8192 entries, each is a bitmask. Precompute per automaton.
    
    # Actually, for BFS we need to apply letters to subsets.
    # Let's use bitmask representation.
    # subset is an int bitmask.
    # Apply letter l: new_subset = OR over s in subset of (1 << trans[l][s])
    
    # Precompute for this automaton: for each letter, a list of length 2^N? No, too big.
    # Instead, precompute for each letter and each state the destination bit.
    dest_bits = []
    for l in range(2):
        db = [1 << trans[l][s] for s in range(N)]
        dest_bits.append(db)
    
    # BFS from full set
    visited = {full: 0}
    queue = deque([full])
    
    while queue:
        subset = queue.popleft()
        dist = visited[subset]
        # Check if singleton
        if subset & (subset - 1) == 0:  # power of 2
            return (dist, True)
        for l in range(2):
            db = dest_bits[l]
            new_subset = 0
            # Apply letter l to subset
            # Iterate over set bits
            temp = subset
            while temp:
                bit = temp & (-temp)
                s = bit.bit_length() - 1
                new_subset |= db[s]
                temp ^= bit
            if new_subset not in visited:
                visited[new_subset] = dist + 1
                queue.append(new_subset)
    
    return (-1, False)
=======
def _compute_threshold(trans):
    """
    Compute exact shortest reset word length via BFS on subset automaton.
    Optimized with precomputed subset transitions.
    trans: list of two lists, trans[letter][state] = dest state.
    Returns (threshold_length, is_synchronizing).
    """
    full = (1 << N) - 1  # bitmask for all states
    num_subsets = 1 << N
    
    # Precompute transitions for all subsets for both letters
    # next_subset[l][mask] = image of mask under letter l
    next_subset = [[0] * num_subsets for _ in range(2)]
    
    for l in range(2):
        # dest_bit[s] is the bitmask for the destination of state s under letter l
        dest_bit = [1 << trans[l][s] for s in range(N)]
        
        for mask in range(num_subsets):
            new_mask = 0
            # Efficiently compute image: iterate set bits
            m = mask
            while m:
                bit = m & (-m)
                s = bit.bit_length() - 1
                new_mask |= dest_bit[s]
                m ^= bit
            next_subset[l][mask] = new_mask
    
    # BFS from full set
    # visited_dist[mask] = distance
    # Use a list for visited distances, initialized to -1
    dist = [-1] * num_subsets
    dist[full] = 0
    queue = deque([full])
    
    while queue:
        subset = queue.popleft()
        d = dist[subset]
        # Check if singleton
        if subset & (subset - 1) == 0:  # power of 2
            return (d, True)
        for l in range(2):
            new_subset = next_subset[l][subset]
            if dist[new_subset] == -1:
                dist[new_subset] = d + 1
                queue.append(new_subset)
    
    return (-1, False)
>>>>>>> REPLACE

