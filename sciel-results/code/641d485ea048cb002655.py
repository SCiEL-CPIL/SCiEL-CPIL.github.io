import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 1
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    nv = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict graph using bitsets
    # adj[i] is a bitmask of valid nodes that conflict with valid_nodes[i]
    adj = [0] * nv
    for i_idx in range(nv):
        v = valid_nodes[i_idx]
        for i_j in range(i_idx + 1, nv):
            u = valid_nodes[i_j]
            if v + u in squares:
                adj[i_idx] |= (1 << i_j)
                adj[i_j] |= (1 << i_idx)
    
    def popcount(x):
        return bin(x).count('1')
        
    def greedy_color_bound(mask):
        """
        Returns an upper bound on the MIS size of the subgraph induced by 'mask'.
        Uses greedy coloring.
        """
        if mask == 0:
            return 0
        
        colors = 0
        remaining = mask
        while remaining:
            colors += 1
            # Pick a vertex with max degree in 'remaining' to help coloring?
            # Or just pick the first one?
            # Picking max degree usually yields fewer colors (tighter bound).
            # To find max degree in 'remaining', we need to iterate.
            # For speed, let's just pick the lowest set bit or highest.
            # Actually, for a tighter bound, we should pick a vertex that conflicts with many others.
            
            # Simple strategy: pick the vertex with the highest index in 'remaining'
            # or the one with the most neighbors in 'remaining'.
            # Calculating degree for all is O(N^2).
            # Let's try a simple heuristic: pick the vertex with the highest degree in the remaining set.
            
            best_v = -1
            max_deg = -1
            temp = remaining
            # Iterate through set bits
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                # Degree of idx in 'remaining'
                deg = popcount(adj[idx] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    best_v = idx
                temp ^= lsb
            
            if best_v == -1:
                break
                
            # Remove best_v and its neighbors from 'remaining'
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
            
        return colors

    best_size = 0
    best_mask = 0
    
    def branch_and_bound(mask, current_size, current_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Upper bound
        ub = greedy_color_bound(mask)
        if current_size + ub <= best_size:
            return
            
        # Select a vertex to branch on
        # Heuristic: pick vertex with max degree in 'mask'
        temp = mask
        best_v = -1
        max_deg = -1
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            deg = popcount(adj[idx] & mask)
            if deg > max_deg:
                max_deg = deg
                best_v = idx
            temp ^= lsb
            
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New mask: remove best_v and its neighbors
        new_mask_include = mask & ~v_bit & ~adj[best_v]
        branch_and_bound(new_mask_include, current_size + 1, current_mask | v_bit)
        
        # Branch 2: Exclude best_v
        new_mask_exclude = mask & ~v_bit
        branch_and_bound(new_mask_exclude, current_size, current_mask)

    # Initial heuristic to set a good lower bound
    # Greedy MIS
    def greedy_mis(order_indices):
        chosen_mask = 0
        for idx in order_indices:
            if (adj[idx] & chosen_mask) == 0:
                chosen_mask |= (1 << idx)
        return chosen_mask

    # Try a few greedy orderings to get a good start
    order_by_degree = sorted(range(nv), key=lambda i: popcount(adj[i]))
    initial_mask = greedy_mis(order_by_degree)
    best_size = popcount(initial_mask)
    best_mask = initial_mask
    
    # Run B&B
    # Start with all valid nodes
    all_mask = (1 << nv) - 1
    branch_and_bound(all_mask, 0, 0)
    
    # If B&B was cut off, we might have a good best_mask.
    # If time remains, do local search on best_mask
    if time.time() < deadline - 5:
        def local_search(mask, max_iters=1000):
            improved = True
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                if time.time() > deadline:
                    break
                # Try adding
                for i in range(nv):
                    if mask & (1 << i):
                        continue
                    if (adj[i] & mask) == 0:
                        mask |= (1 << i)
                        improved = True
                # Try swapping
                for i in range(nv):
                    if mask & (1 << i):
                        continue
                    conf = adj[i] & mask
                    if conf == 0:
                        mask |= (1 << i)
                        improved = True
                        continue
                    temp = conf
                    while temp:
                        r_bit = temp & (-temp)
                        r_idx = r_bit.bit_length() - 1
                        if (adj[i] & (mask ^ r_bit)) == 0:
                            mask = (mask & ~r_bit) | (1 << i)
                            improved = True
                            break
                        temp ^= r_bit
            return mask

        mask = local_search(best_mask)
        if popcount(mask) > best_size:
            best_size = popcount(mask)
            best_mask = mask

    # Convert to list
    result = []
    for i in range(nv):
        if best_mask & (1 << i):
            result.append(valid_nodes[i])
            
    if not result:
        result = [valid_nodes[0]]
        
    return result
