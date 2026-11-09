import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: Andrasfai graph on 26 vertices (alpha=9, triangle-free)
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # Try to find larger graphs, descending from 40
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        found = False
        restarts = 0
        max_restarts = 50000
        
        while time.time() < deadline - 3 and restarts < max_restarts and not found:
            restarts += 1
            
            # Generate a random maximal triangle-free graph
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            all_mask = (1 << n) - 1
            non_adj = [0] * n
            for i in range(n):
                non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
            # Try to reduce alpha by adding edges (maintaining triangle-free)
            # Greedily add edges that don't create triangles
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try to add a random non-edge that doesn't create a triangle
                missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                          if not (adj[i] >> j) & 1]
                if not missing:
                    break
                random.shuffle(missing)
                for u, v in missing[:100]:  # Try up to 100 random missing edges
                    if (adj[u] & adj[v]) == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        non_adj[u] = (all_mask ^ (1 << u)) & ~adj[u]
                        non_adj[v] = (all_mask ^ (1 << v)) & ~adj[v]
                        improved = True
                        break
            
            # Check if alpha < 10
            if not exact_has_independent_set_10(adj, n, all_mask, non_adj):
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

def exact_has_independent_set_10(adj, n, all_mask, non_adj):
    """
    Returns True if there exists an independent set of size 10.
    Uses branch and bound with strong pruning.
    """
    # Precompute degrees for heuristic
    def dfs(mask, size):
        if size == 10:
            return True
        if mask == 0:
            return False
        remaining = mask.bit_count()
        if remaining + size < 10:
            return False
        # Upper bound: greedy coloring of the subgraph induced by mask
        # If the clique cover number of the complement (i.e., coloring of induced subgraph)
        # is less than 10 - size, then no independent set of size 10 - size exists
        # Simpler bound: if remaining < 10 - size, fail (already covered)
        
        # Heuristic: pick vertex with maximum degree in the induced subgraph
        max_deg = -1
        best_v = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return False
        
        v_bit = 1 << best_v
        
        # Include best_v: next candidates are non-neighbors of best_v in mask
        if dfs(non_adj[best_v] & mask, size + 1):
            return True
        
        # Exclude best_v
        if dfs(mask & ~v_bit, size):
            return True
        
        return False
    
    return dfs(all_mask, 0)
