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
        all_mask = (1 << n) - 1
        
        while time.time() < deadline - 5:  # Keep 5s buffer
            restarts += 1
            if restarts > 10000:  # Cap restarts
                break
                
            # Generate a random maximal triangle-free graph using a faster method
            # adj[i] is a bitmask of neighbors of i
            adj = [0] * n
            
            # Fast random maximal triangle-free generation:
            # For each vertex, randomly decide to connect to non-conflicting vertices
            # Process vertices in random order, adding edges greedily
            order = list(range(n))
            random.shuffle(order)
            
            for u in order:
                # Get all potential neighbors not yet processed or already processed
                # To ensure triangle-free: when adding edge (u,v), check no common neighbor
                # We process in random order, so for each u, consider all v > u in some sense
                # Actually, let's use a different approach: random permutation, add edges greedily
                
                # For vertex u, consider all other vertices v
                # We can add edge (u,v) if no triangle is formed
                # Triangle forms if there exists w such that (u,w) and (v,w) are both edges
                # So we need adj[u] & adj[v] == 0 before adding (u,v)
                
                # To make it efficient, process in the shuffled order
                # When processing u, we can add edges to all v that don't create triangles
                candidates = []
                for v in range(n):
                    if v == u:
                        continue
                    # Check if adding (u,v) creates a triangle
                    if adj[u] & adj[v] == 0:
                        candidates.append(v)
                
                # Randomly add a subset of candidate edges
                # To make it "maximal", we should add all possible edges
                # But that might be too dense. Let's add them with some probability
                # Actually for triangle-free maximal graphs, we should add all non-conflicting edges
                for v in candidates:
                    # Only add if v > u to avoid duplicate processing
                    # But we're processing in random order, so just add if not already present
                    if not (adj[u] & (1 << v)):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Quick check: is the graph triangle-free? (should be by construction)
            # Check if alpha < 10 using branch and bound
            
            # Precompute non-neighbor masks
            non_neighbors = [0] * n
            for i in range(n):
                all_except_i = all_mask ^ (1 << i)
                non_neighbors[i] = all_except_i & ~adj[i]
            
            # Branch and bound for independent set of size 10
            def has_independent_set_10():
                # Use iterative deepening / branch and bound
                # Key insight: if we can find ANY independent set of size 10, return True
                # We want to know if NO such set exists
                
                # Use memoization with a set of visited states (mask, size) - but size is determined by mask in some sense
                # Actually, let's use a simple recursive B&B with good pruning
                
                # First, do a greedy check: if greedy can't find size 10, still need B&B
                # But if greedy finds size 10, we can return True immediately
                
                # Greedy independent set
                def greedy_independent_set(mask, target):
                    result = []
                    m = mask
                    while m and len(result) < target:
                        # Pick vertex with minimum degree in induced subgraph
                        # Or just pick first for speed
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        result.append(v)
                        m &= ~non_neighbors[v]
                        m &= ~(1 << v)
                    return len(result) >= target
                
                if greedy_independent_set(all_mask, 10):
                    return True
                
                # Now do B&B to prove non-existence
                # State: (mask, remaining) where remaining = 10 - current_size
                # Use recursion with strong pruning
                
                # Precompute: for each mask, what's the max independent set size?
                # Too expensive. Instead, use upper bound: popcount(mask)
                
                def dfs(mask, remaining):
                    if remaining == 0:
                        return True
                    if mask == 0:
                        return False
                    if mask.bit_count() < remaining:
                        return False
                    
                    # Upper bound: can we even fit 'remaining' vertices?
                    # Simple bound: popcount(mask) >= remaining (already checked)
                    # Better bound: use the fact that in the induced subgraph,
                    # alpha <= n - minimum vertex cover, but that's complex
                    
                    # Pick vertex with minimum non-neighbor count in mask (most constrained)
                    min_cnt = n + 1
                    best_v = -1
                    temp = mask
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        cnt = (non_neighbors[v] & mask).bit_count()
                        if cnt < min_cnt:
                            min_cnt = cnt
                            best_v = v
                        temp ^= lsb
                    
                    if best_v == -1:
                        return False
                    
                    v_bit = 1 << best_v
                    
                    # Include best_v
                    new_mask_inc = non_neighbors[best_v] & mask
                    new_mask_inc &= ~v_bit
                    if dfs(new_mask_inc, remaining - 1):
                        return True
                    
                    # Exclude best_v
                    new_mask_exc = mask & ~v_bit
                    if dfs(new_mask_exc, remaining):
                        return True
                    
                    return False
                
                return dfs(all_mask, 10)
            
            if not has_independent_set_10():
                # Valid graph found
                edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
        
    return best_result
