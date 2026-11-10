import time
import random

def construct():
    start_time = time.time()
    # Safety margin: kill at 55s, limit is 60s
    deadline = start_time + 55.0
    
    N = 256
    
    # 1. Identify valid nodes (those that don't have self-conflicts)
    # A node 'a' is invalid if 2*a is a perfect square
    valid_nodes = []
    for a in range(1, N + 1):
        s = 2 * a
        r = int(s**0.5)
        if r * r != s:
            valid_nodes.append(a)
            
    # Map valid node values to indices 0..M-1
    val_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    # 2. Precompute neighbor masks
    # neighbor_masks[i] is a bitmask of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    # Note: We only care about conflicts between distinct valid nodes.
    # If a+b is a square, they conflict.
    squares = set(i*i for i in range(1, int((2*N)**0.5) + 2))
    
    neighbor_masks = [0] * M
    
    for i, a in enumerate(valid_nodes):
        mask = 0
        for j, b in enumerate(valid_nodes):
            if i == j:
                continue
            if a + b in squares:
                mask |= (1 << j)
        neighbor_masks[i] = mask
        
    # 3. Search for Maximum Independent Set
    best_size = 0
    best_mask = 0
    
    # Greedy initialization with random ordering
    # Using bitmasks makes the conflict check O(1)
    
    # To optimize the shuffle, we can pre-generate random permutations or just use random.shuffle
    # Given M <= 256, shuffle is fast.
    
    while time.time() < deadline:
        indices = list(range(M))
        random.shuffle(indices)
        
        current_mask = 0
        count = 0
        
        for idx in indices:
            # Check if node idx conflicts with any node in current_mask
            # If neighbor_masks[idx] & current_mask == 0, no conflict
            if (neighbor_masks[idx] & current_mask) == 0:
                current_mask |= (1 << idx)
                count += 1
                
        if count > best_size:
            best_size = count
            best_mask = current_mask
            
    # Convert best_mask back to list of values
    result = []
    for i, v in enumerate(valid_nodes):
        if (best_mask >> i) & 1:
            result.append(v)
            
    # If for some reason result is empty (shouldn't happen), return a single valid node
    if not result and valid_nodes:
        result = [valid_nodes[0]]
        
    return result
