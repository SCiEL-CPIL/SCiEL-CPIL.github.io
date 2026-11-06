import time

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin for 60s limit
    
    n = 256
    max_sum = 2 * n
    
    # Identify squares
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # Filter valid numbers (2*a not square)
    valid_numbers = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_numbers.append(a)
            
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict masks (neighbors)
    # neighbors[i] is a bitmask of indices j such that valid_numbers[i] + valid_numbers[j] is a square
    neighbors = [0] * num_valid
    
    for i in range(num_valid):
        a = valid_numbers[i]
        mask = 0
        for s in squares:
            b = s - a
            if b in val_to_idx:
                j = val_to_idx[b]
                if i != j: # We already handled i==j by filtering valid_numbers
                    mask |= (1 << j)
        neighbors[i] = mask
        
    # Initial greedy to get a good lower bound
    # Sort by degree descending
    degrees = [bin(neighbors[i]).count('1') for i in range(num_valid)]
    order = sorted(range(num_valid), key=lambda x: degrees[x], reverse=True)
    
    current_best_set = 0
    current_best_size = 0
    remaining = (1 << num_valid) - 1
    
    for v in order:
        if remaining & (1 << v):
            # Check if v is compatible with current_best_set
            # A vertex v is compatible if it has no neighbors in current_best_set
            # i.e., neighbors[v] & current_best_set == 0
            if neighbors[v] & current_best_set == 0:
                current_best_set |= (1 << v)
                current_best_size += 1
                # Remove v and its neighbors from remaining
                remaining &= ~(neighbors[v] | (1 << v))
                
    best_size = current_best_size
    best_mask = current_best_set
    
    # Branch and Bound
    # State: candidates (bitmask), current_mask (bitmask), current_size (int)
    def solve(candidates, current_mask, current_size):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        # Prune
        if current_size + candidates.bit_count() <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Heuristic: pick vertex with max degree in candidates
        # To avoid O(N^2) in Python, we can just pick the first one or a random one?
        # Better: iterate through bits in candidates and find one with max overlap.
        # Since N is small (<=256), this is fine.
        
        max_deg = -1
        chosen_v = -1
        c = candidates
        while c:
            # Get lowest set bit
            lsb = c & -c
            v = lsb.bit_length() - 1
            # Degree in candidates
            deg = (candidates & neighbors[v]).bit_count()
            if deg > max_deg:
                max_deg = deg
                chosen_v = v
            c ^= lsb
            
        if chosen_v == -1:
            return
            
        # Branch 1: Include chosen_v
        # New candidates: candidates without chosen_v and without its neighbors
        new_candidates_include = candidates & ~(neighbors[chosen_v] | (1 << chosen_v))
        new_mask_include = current_mask | (1 << chosen_v)
        solve(new_candidates_include, new_mask_include, current_size + 1)
        
        # Branch 2: Exclude chosen_v
        new_candidates_exclude = candidates & ~(1 << chosen_v)
        solve(new_candidates_exclude, current_mask, current_size)

    # Initial candidates is all valid numbers
    all_candidates = (1 << num_valid) - 1
    solve(all_candidates, 0, 0)
    
    # Extract result
    result = []
    for i in range(num_valid):
        if best_mask & (1 << i):
            result.append(valid_numbers[i])
            
    if not result:
        # Fallback, though unlikely
        result = [1]
        
    return result
