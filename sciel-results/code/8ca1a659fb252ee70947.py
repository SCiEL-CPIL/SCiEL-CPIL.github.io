import random
import time

def construct():
    N = 256
    nodes = list(range(1, N + 1))
    
    # Precompute adjacency masks
    # adj_mask[i] is an integer where bit j is 1 if i and j conflict
    adj_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                mask |= (1 << j)
                adj_mask[j] |= (1 << i)
        adj_mask[i] = mask

    def greedy_independent_set(order):
        selected_mask = 0
        count = 0
        for u in order:
            # Check if u conflicts with any selected node
            if (adj_mask[u] & selected_mask) == 0:
                selected_mask |= (1 << u)
                count += 1
        return count, selected_mask

    best_count = 0
    best_mask = 0
    
    start_time = time.time()
    timeout = 50.0 # Safety margin
    
    # 1. Deterministic orders
    # Ascending
    c, m = greedy_independent_set(nodes)
    if c > best_count:
        best_count = c
        best_mask = m
        
    # Descending
    c, m = greedy_independent_set(sorted(nodes, reverse=True))
    if c > best_count:
        best_count = c
        best_mask = m
        
    # Degree based (ascending degree)
    degrees = [bin(adj_mask[i]).count('1') for i in range(N + 1)]
    order_by_degree = sorted(nodes, key=lambda x: degrees[x])
    c, m = greedy_independent_set(order_by_degree)
    if c > best_count:
        best_count = c
        best_mask = m

    # 2. Random restarts
    # With bitmask, each greedy pass is very fast.
    # We can do thousands of them.
    while time.time() - start_time < timeout:
        random_order = nodes[:]
        random.shuffle(random_order)
        c, m = greedy_independent_set(random_order)
        if c > best_count:
            best_count = c
            best_mask = m

    # Convert mask back to list
    result = []
    for i in range(1, N + 1):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result
