import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave margin for evaluation overhead
    
    # Fallback: Andrasfai graph And(9) on 26 vertices
    # Triangle-free, alpha = 9
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this n
        restarts = 0
        while time.time() < deadline - 5:  # Keep 5s buffer
            restarts += 1
            if restarts > 5000: # Cap restarts
                break
                
            # Generate a random maximal triangle-free graph
            # adj[i] is a bitmask of neighbors of i
            adj = [0] * n
            
            # Generate all pairs and shuffle
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                # Check if adding (u, v) creates a triangle
                # Triangle exists if there is a common neighbor
                common = adj[u] & adj[v]
                if common == 0:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if alpha < 10
            # We use a branch and bound to see if an independent set of size 10 exists.
            # If it does NOT exist, the graph is valid.
            
            # Precompute non-neighbor masks
            # non_neighbors[i] contains bits for j != i such that not (adj[i] & (1<<j))
            non_neighbors = [0] * n
            all_mask = (1 << n) - 1
            for i in range(n):
                # All vertices except i
                all_except_i = all_mask ^ (1 << i)
                # Non-neighbors are those in all_except_i that are NOT in adj[i]
                non_neighbors[i] = all_except_i & ~adj[i]
            
            # Branch and bound for independent set of size 10
            # We want to know if there is an independent set of size 10.
            # If we return False, it means no such set exists, so alpha < 10.
            
            def has_independent_set_10():
                # State: mask of available vertices, current size of independent set
                # We use recursion with pruning
                
                # Precompute popcount for masks? No, just use bin(x).count('1') or int.bit_count()
                
                def dfs(mask, size):
                    if size == 10:
                        return True
                    if mask == 0:
                        return False
                    # Pruning: if number of bits in mask < 10 - size, impossible
                    if mask.bit_count() < 10 - size:
                        return False
                    
                    # Pick a vertex from mask
                    # Heuristic: pick vertex v in mask with minimum popcount(non_neighbors[v] & mask)
                    # This is the "most constrained" vertex.
                    # Finding this takes O(n) per step.
                    
                    # For speed, we might just pick the lowest set bit if n is small.
                    # But the heuristic is important for proving non-existence.
                    
                    # Let's try the heuristic if mask has > 15 bits, otherwise pick first.
                    if mask.bit_count() > 15:
                        min_count = n + 1
                        best_v = -1
                        temp_mask = mask
                        while temp_mask:
                            lsb = temp_mask & -temp_mask
                            v = lsb.bit_length() - 1
                            # Count non-neighbors of v in mask
                            cnt = (non_neighbors[v] & mask).bit_count()
                            if cnt < min_count:
                                min_count = cnt
                                best_v = v
                            temp_mask ^= lsb
                    else:
                        # Just pick the lowest set bit
                        lsb = mask & -mask
                        best_v = lsb.bit_length() - 1
                    
                    if best_v == -1:
                        return False
                    
                    v_bit = 1 << best_v
                    
                    # Option 1: Include best_v
                    # New mask: non-neighbors of best_v within current mask, excluding best_v itself
                    new_mask_include = non_neighbors[best_v] & mask
                    
                    if dfs(new_mask_include, size + 1):
                        return True
                    
                    # Option 2: Exclude best_v
                    new_mask_exclude = mask & ~v_bit
                    if dfs(new_mask_exclude, size):
                        return True
                    
                    return False
                
                return dfs(all_mask, 0)
            
            if not has_independent_set_10():
                # Valid graph found
                edges = []
                for i in range(n):
                    mask = adj[i]
                    # Only add edges where i < j to avoid duplicates
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
        
    return best_result
