import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave 35s for safety/overhead
    
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
        max_restarts = 5000
        
        while time.time() < deadline - 5 and restarts < max_restarts:
            restarts += 1
            
            # Generate random maximal triangle-free graph
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if alpha < 10 using a more efficient branch and bound
            all_mask = (1 << n) - 1
            non_adj = [0] * n
            for i in range(n):
                non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
            # Precompute degree for heuristic
            degrees = [0] * n
            for i in range(n):
                degrees[i] = adj[i].bit_count()
            
            # Use a faster independent set checker
            # We want to check if there is an independent set of size 10
            # If max independent set < 10, we're good
            
            # Greedy upper bound: repeatedly pick vertex with min degree in remaining set
            # But we need exact check. Let's use a simple DFS with better pruning.
            
            def has_independent_set_10():
                # Use branch and bound
                # State: current mask of available vertices, current independent set size
                # We want to find if we can reach size 10
                
                # Simple approach: try to build independent set greedily, then backtrack
                # But let's use a recursive search with memoization-like pruning
                
                def dfs(mask, size):
                    if size == 10:
                        return True
                    if mask == 0:
                        return False
                    
                    # Pruning: if remaining vertices + current size < 10, fail
                    if mask.bit_count() + size < 10:
                        return False
                    
                    # Heuristic: pick vertex with minimum degree in mask (for independent set,
                    # we want to minimize constraints, so pick vertex with fewest non-neighbors?
                    # Actually, for finding independent set, we should pick a vertex and
                    # either include it (remove its neighbors) or exclude it.
                    # To find if ANY independent set of size 10 exists, we can use:
                    # - If we include v, we can only use non-neighbors of v
                    # - If we exclude v, we use mask without v
                    
                    # Better heuristic: pick vertex with MAX degree in the complement
                    # (i.e., min degree in original graph restricted to mask)
                    # This way, when we include it, we remove many vertices,
                    # which helps prune the "exclude" branch less but the "include" branch more.
                    
                    # Actually, standard heuristic for independent set:
                    # Pick vertex with maximum degree in the graph induced by mask.
                    # When included, its neighbors are removed (good for pruning include branch).
                    # When excluded, only it is removed.
                    
                    max_deg = -1
                    best_v = -1
                    temp = mask
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        # Degree of v in the graph induced by mask
                        deg = (adj[v] & mask).bit_count()
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                        temp ^= lsb
                    
                    if best_v == -1:
                        return False
                    
                    v_bit = 1 << best_v
                    
                    # Try including best_v: new mask is non-neighbors of best_v in mask
                    include_mask = non_adj[best_v] & mask
                    if dfs(include_mask, size + 1):
                        return True
                    
                    # Try excluding best_v
                    exclude_mask = mask & ~v_bit
                    if dfs(exclude_mask, size):
                        return True
                    
                    return False
                
                return dfs(all_mask, 0)
            
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
