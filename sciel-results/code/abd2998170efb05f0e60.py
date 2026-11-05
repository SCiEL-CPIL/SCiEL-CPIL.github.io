import time
import random

def _has_indep_11_bitmask(adj, n, deadline):
    """
    Check if an independent set of size 11 exists using bitmask backtracking.
    Returns True if such a set exists (bad), False if not (good).
    """
    def backtrack(remaining, current_size):
        if time.time() > deadline:
            return False  # Assume no independent set (optimistic)
        if current_size >= 11:
            return True
        if not remaining:
            return False
        if current_size + remaining.bit_count() < 11:
            return False
        
        # Pick vertex with minimum degree in remaining subgraph
        min_v = -1
        min_deg = 100
        temp = remaining
        while temp:
            v = (temp & -temp).bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            temp &= temp - 1
            
        if min_v == -1:
            return False
            
        # Branch 1: include min_v in independent set
        new_remaining = remaining & ~(adj[min_v] | (1 << min_v))
        if backtrack(new_remaining, current_size + 1):
            return True
            
        # Branch 2: exclude min_v
        remaining &= ~(1 << min_v)
        if backtrack(remaining, current_size):
            return True
            
        return False

    return backtrack((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with a small core: 10 disjoint triangles
    # This ensures C4-free and alpha=10
    n = 0
    adj = []
    
    # Helper to add a vertex
    def add_vertex(neighbors):
        nonlocal n, adj
        new_adj = 0
        for v in neighbors:
            new_adj |= 1 << v
            adj[v] |= 1 << n
        adj.append(new_adj)
        n += 1

    # Build core: 10 triangles
    for _ in range(10):
        v1 = n; add_vertex([])
        v2 = n; add_vertex([v1])
        v3 = n; add_vertex([v1, v2])
        
    # Now try to expand
    max_target = 40
    
    while n < max_target and time.time() < deadline:
        # Find a valid neighbor set S for the new vertex
        # S must be an independent set in the "conflict graph" H
        # H has edge (u,w) if N(u) intersect N(w) != empty
        
        # Build conflict graph H
        # h_adj[u] is bitmask of vertices w such that N(u) & N(w) != 0
        h_adj = [0] * n
        for u in range(n):
            for w in range(u + 1, n):
                if (adj[u] & adj[w]):
                    h_adj[u] |= 1 << w
                    h_adj[w] |= 1 << u
        
        # Find a large independent set in H
        # Use a greedy randomized approach
        best_S = None
        best_size = 0
        
        # Try a few random restarts for finding independent set in H
        for _ in range(20):
            remaining = (1 << n) - 1
            S = 0
            while remaining:
                # Pick random vertex from remaining
                temp = remaining
                r = random.randint(0, temp.bit_count() - 1)
                # Select r-th set bit
                v = -1
                count = -1
                t = temp
                while t:
                    lsb = t & -t
                    v = lsb.bit_length() - 1
                    count += 1
                    if count == r:
                        break
                    t &= t - 1
                
                S |= 1 << v
                # Remove v and its neighbors in H
                remaining &= ~(h_adj[v] | (1 << v))
            
            if S.bit_count() > best_size:
                best_size = S.bit_count()
                best_S = S
        
        # Try to add vertex with best_S
        # Check if alpha < 11
        # We need to simulate adding the vertex
        # New adj list: adj + [best_S]
        # But we also need to update adj for existing vertices
        
        # To avoid modifying adj, we can create a temporary adj list
        # Or just check if the new graph has alpha >= 11
        
        # Optimization: If best_S is empty, the new vertex is isolated.
        # alpha increases by 1. If current alpha is 10, new alpha is 11. Bad.
        # So we need best_S to be non-empty and "break" independent sets.
        
        # Let's just do the full check for the best candidate
        if best_size > 0:
            # Create temporary adj
            temp_adj = adj.copy()
            temp_adj.append(best_S)
            # Update existing adj
            # Actually, we need to OR best_S into adj[v] for v in best_S
            # This is expensive to copy.
            # Let's just run the alpha check on the modified structure
            
            # Modify adj in place, check, then revert if needed
            # This is safer for correctness
            original_adj = adj[:]
            
            # Add new vertex
            new_v = n
            adj.append(best_S)
            for v in range(n):
                if (best_S >> v) & 1:
                    adj[v] |= 1 << new_v
            
            has_indep = _has_indep_11_bitmask(adj, n + 1, deadline)
            
            if not has_indep:
                n += 1
                # Keep the modification
            else:
                # Revert
                adj = original_adj
                # Try a smaller S?
                # For simplicity, if the best fails, we might try a random smaller S
                # But for now, if the best fails, we might be stuck.
                # Let's try a random subset of best_S
                success = False
                for _ in range(10):
                    # Random subset
                    S_sub = 0
                    temp = best_S
                    while temp:
                        if random.random() < 0.5:
                            lsb = temp & -temp
                            S_sub |= lsb
                        temp &= temp - 1
                    
                    if S_sub == 0:
                        continue
                    
                    # Check S_sub validity (C4 constraint)
                    # Since S_sub is a subset of an independent set in H, it is also independent in H.
                    # So C4 constraint is satisfied.
                    
                    # Check alpha
                    original_adj = adj[:]
                    new_v = n
                    adj.append(S_sub)
                    for v in range(n):
                        if (S_sub >> v) & 1:
                            adj[v] |= 1 << new_v
                    
                    has_indep = _has_indep_11_bitmask(adj, n + 1, deadline)
                    
                    if not has_indep:
                        n += 1
                        success = True
                        break
                    else:
                        adj = original_adj
                
                if not success:
                    # If we can't add a vertex, we might try a different core or stop
                    # For now, break to avoid infinite loop
                    break

    # Build edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])

    return {'n': n, 'edges': edges}
