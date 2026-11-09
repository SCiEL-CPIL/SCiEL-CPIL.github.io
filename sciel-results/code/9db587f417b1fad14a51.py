import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # 1. Precompute squares
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # 2. Filter self-conflicts
    # If 2*v is a square, v cannot be in the set
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    M = len(valid_nodes)
    
    # 3. Build conflict masks
    # conflict_mask[i] is a bitmask of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    conflict_mask = [0] * M
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    best_size = 0
    best_mask = 0
    
    # Local search using bitmasks
    def local_search(init_mask):
        S_mask = init_mask
        current_size = bin(S_mask).count('1')
        
        while True:
            improved = False
            
            # 1. Greedy Add
            for i in range(M):
                if (S_mask >> i) & 1:
                    continue
                # Check conflict
                if (conflict_mask[i] & S_mask) == 0:
                    S_mask |= (1 << i)
                    current_size += 1
                    improved = True
            
            if not improved:
                # 2. Try Swap (Remove one, Add many)
                # Iterate through nodes in S
                for i in range(M):
                    if not ((S_mask >> i) & 1):
                        continue
                    
                    # Remove i
                    S_mask_no_i = S_mask & ~(1 << i)
                    size_after_remove = current_size - 1
                    
                    # Find all nodes that can be added
                    to_add_mask = 0
                    for j in range(M):
                        if j == i:
                            continue
                        if (S_mask_no_i >> j) & 1:
                            continue
                        if (conflict_mask[j] & S_mask_no_i) == 0:
                            to_add_mask |= (1 << j)
                    
                    num_adds = bin(to_add_mask).count('1')
                    if num_adds > 0:
                        new_size = size_after_remove + num_adds
                        if new_size > current_size:
                            S_mask = S_mask_no_i | to_add_mask
                            current_size = new_size
                            improved = True
                            break # Restart loop with new mask
                
                if not improved:
                    break
        
        return S_mask, current_size

    # Multiple restarts
    indices = list(range(M))
    
    # Run a few greedy passes with random orders
    # 1000 restarts should be fast enough with bitmasks
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        random.shuffle(indices)
        
        init_mask = 0
        for idx in indices:
            if (conflict_mask[idx] & init_mask) == 0:
                init_mask |= (1 << idx)
        
        S_mask, size = local_search(init_mask)
        
        if size > best_size:
            best_size = size
            best_mask = S_mask
    
    # Convert best_mask back to list of nodes
    result = []
    for i in range(M):
        if (best_mask >> i) & 1:
            result.append(valid_nodes[i])
    
    return result
