import time
import random

def construct():
    n = 16
    deadline = time.time() + 110  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # --- Helpers ---
    
    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        color = [-1] * n
        # Order vertices by degree descending to improve pruning
        # We need to recompute this if adj changes, but for a single check it's fine.
        # However, doing this every call is expensive. 
        # Let's just use a static order or simple 0..n-1 first, 
        # but degree ordering is much better.
        # For n=16, even simple ordering is likely fast enough if the graph is dense.
        # Let's try static ordering 0..15 first for speed, 
        # but dynamic ordering is better for correctness of pruning.
        # Given n=16, let's do dynamic ordering.
        
        order = sorted(range(n), key=lambda x: -adj[x].bit_count())
        
        def backtrack(k):
            if k == n:
                return True
            u = order[k]
            used_colors = 0
            mask = adj[u]
            # Check neighbors
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if color[v] != -1:
                    used_colors |= (1 << color[v])
                m ^= lsb
                
            for c in range(3):
                if not (used_colors & (1 << c)):
                    color[u] = c
                    if backtrack(k + 1):
                        return True
                    color[u] = -1
            return False
            
        return backtrack(0)

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            # Only consider v > u
            mask_higher = mask & ~((1 << (u + 1)) - 1)
            m = mask_higher
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                edges.append([u, v])
                m ^= lsb
        return edges

    def make_grotzsch_adj():
        adj = [0] * n
        # C5: 0-1-2-3-4-0
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Copies 5-9 for 0-4
        for u, v in cycle:
            # (u, v') => (u, 5+v)
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            # (v, u') => (v, 5+u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
            
        # w=10 connected to 5-9
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
            
        return adj

    # --- Main Search Loop ---
    
    while time.time() < deadline:
        # Start with Grötzsch
        adj = make_grotzsch_adj()
        
        # Check if initial is valid (it is, Grötzsch is 4-chromatic)
        if is_3_colorable(adj):
            continue # Should not happen
        
        current_edges = get_edges_from_adj(adj)
        current_count = len(current_edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges
        
        # Local Search
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add edges
            possible_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_triangle_free_add(adj, u, v):
                            possible_edges.append((u, v))
            
            random.shuffle(possible_edges)
            
            for u, v in possible_edges:
                if time.time() > deadline:
                    break
                
                # Tentatively add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not is_3_colorable(adj):
                    # Accept
                    improved = True
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = get_edges_from_adj(adj)
                    break # Restart scan after adding one edge
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if improved:
                continue
            
            # 2. If no edge could be added, try to remove one to escape local optimum
            # We want to remove an edge that is "non-essential" for the 4-chromatic property
            # or just any edge, hoping that adding different edges later yields a better graph.
            
            # Get list of existing edges
            existing_edges = []
            for u in range(n):
                mask = adj[u]
                mask_higher = mask & ~((1 << (u + 1)) - 1)
                m = mask_higher
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    existing_edges.append((u, v))
                    m ^= lsb
            
            if not existing_edges:
                break # Empty graph, stuck
                
            random.shuffle(existing_edges)
            
            removed = False
            for u, v in existing_edges:
                if time.time() > deadline:
                    break
                
                # Tentatively remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Check if still non-3-colorable
                if not is_3_colorable(adj):
                    # Accept removal (we lose 1 edge, but might gain more later)
                    current_count -= 1
                    removed = True
                    break
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not removed:
                # Cannot remove any edge without becoming 3-colorable.
                # This is a local minimum where we cannot add or remove.
                # Break inner loop and restart with a new seed or different strategy.
                break
                
    return best_edges
