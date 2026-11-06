import time
import random
import math

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy:
    1. Start with a small valid C4-free graph.
    2. Use simulated annealing to grow the graph by adding vertices.
    3. Maintain C4-freedom as a hard constraint.
    4. Minimize independence number while maximizing vertex count.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    def popcount(x):
        return bin(x).count('1')
    
    def has_c4(n, adj):
        """Check if graph has a C4 using bitmasks."""
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if popcount(common) >= 2:
                    return True
        return False
    
    def can_add_edge_fast(u, v, adj, n):
        """
        Check if adding edge (u, v) creates a C4.
        Fast version using bitmask operations.
        """
        mask_u = adj[u]
        mask_v = adj[v]
        
        # Check if any neighbor of u is adjacent to any neighbor of v
        # This would create a C4: u - a - b - v
        m = mask_u
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m -= lsb
            if adj[a] & mask_v:
                return False
        return True
    
    def max_independent_set_size(n, adj, deadline):
        """
        Find the size of the maximum independent set using branch and bound.
        """
        best = [0]
        
        def branch(candidates_mask, current_size):
            if time.time() > deadline - 3:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Pick a vertex with high degree in candidate subgraph for better pruning
            # Find vertex with max degree in candidates
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m -= lsb
                deg = popcount(adj[v] & candidates_mask)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v == -1:
                return
            v = best_v
            
            # Option 1: Include v
            new_candidates = candidates_mask & ~adj[v] & ~(1 << v)
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude v
            branch(candidates_mask & ~(1 << v), current_size)

        branch((1 << n) - 1, 0)
        return best[0]
    
    def verify_graph(n, adj, deadline):
        """Verify C4-free and alpha < 11."""
        if has_c4(n, adj):
            return False
        alpha = max_independent_set_size(n, adj, deadline)
        return alpha < 11
    
    # Start with a small valid graph: 10 disjoint triangles (n=30)
    # This is C4-free and has alpha = 10 (one from each triangle)
    n = 30
    adj = [0] * n
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    
    # Verify initial graph
    if not verify_graph(n, adj, deadline):
        # Should not happen, but fallback
        return {'n': 30, 'edges': []}
    
    best_n = n
    best_adj = adj[:]
    
    # Try to grow the graph using simulated annealing
    # We'll try to add vertices one at a time
    
    current_n = n
    current_adj = adj[:]
    current_alpha = 10
    
    for target_n in range(31, 44):
        if time.time() > deadline - 10:
            break
        
        # Try to add a new vertex
        added = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            random.seed(42 + attempt + target_n * 1000)
            
            # Create new graph with target_n vertices
            new_adj = current_adj[:] + [0]
            new_n = target_n
            
            # Try to connect the new vertex to some existing vertices
            # We want to add edges that don't create C4 and help reduce alpha
            
            # Randomly choose which vertices to connect to
            possible_connections = list(range(new_n - 1))
            random.shuffle(possible_connections)
            
            # Try different numbers of connections
            for num_edges in [1, 2, 3, 4]:
                if time.time() > deadline - 5:
                    break
                
                chosen = possible_connections[:num_edges]
                
                # Try to add edges to new vertex
                valid_addition = True
                temp_adj = new_adj[:]
                
                for v in chosen:
                    if not can_add_edge_fast(new_n - 1, v, temp_adj, new_n):
                        valid_addition = False
                        break
                    temp_adj[new_n - 1] |= (1 << v)
                    temp_adj[v] |= (1 << (new_n - 1))
                
                if not valid_addition:
                    continue
                
                # Check if this graph is valid
                if verify_graph(new_n, temp_adj, deadline):
                    alpha = max_independent_set_size(new_n, temp_adj, deadline)
                    if alpha < 11:
                        current_n = new_n
                        current_adj = temp_adj
                        current_alpha = alpha
                        added = True
                        break
            
            if added:
                break
        
        if added:
            best_n = current_n
            best_adj = current_adj[:]
        else:
            # Could not add vertex, stop
            break
    
    # Convert best_adj to edge list
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])
    
    return {'n': best_n, 'edges': edge_list}
