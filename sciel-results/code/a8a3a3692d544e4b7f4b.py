def _evaluate(transitions):
    # transitions: list of two lists of 12 ints
    # Precompute mask transitions? No, just compute on the fly.
    # N=12, full_set = (1<<12)-1
    full_set = (1 << N) - 1
    visited = {full_set}
    queue = deque([(full_set, 0)])
    
    # Precompute transition masks for each letter?
    # trans[0][i] is dest of i.
    # To apply letter l to mask m:
    # new_mask = 0
    # for i in range(N):
    #   if m & (1<<i):
    #     new_mask |= 1 << transitions[l][i]
    
    # We can precompute the effect of each letter on each bit?
    # Or just loop. 12 bits is small.
    
    while queue:
        mask, depth = queue.popleft()
        if mask.bit_count() == 1: # Python 3.8+? No, use bin(mask).count('1') or precomputed popcount
            return depth
        if depth > 150: # Cap
            return -1 # Not synchronizing or too deep
        for l in range(2):
            new_mask = 0
            t = transitions[l]
            for i in range(N):
                if mask & (1 << i):
                    new_mask |= (1 << t[i])
            if new_mask not in visited:
                visited.add(new_mask)
                queue.append((new_mask, depth + 1))
    return -1
