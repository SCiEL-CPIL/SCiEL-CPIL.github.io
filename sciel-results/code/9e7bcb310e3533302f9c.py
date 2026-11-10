import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(all_edges)

    best_edges = 0
    best_graph = set()

    # Precompute neighbor masks (bitmasks) for fast iteration
    # adj[i] is a bitmask of neighbors of vertex i
    def creates_q3(adj, u, v):
        """
        Check if adding edge (u,v) to the graph creates a Q3 subgraph.
        We assume (u,v) is NOT yet in adj.
        
        Q3 structure: A cube has 8 vertices. If we fix an edge (u,v),
        the cube consists of:
        - Edge (u,v)
        - Two squares sharing (u,v):
          Square 1: u-v-a-b-u  (a~v, b~u, a~b)
          Square 2: u-v-c-d-u  (c~v, d~u, c~d)
          where {a,b} and {c,d} are disjoint sets of 2 vertices each.
        - Two remaining vertices e, f such that:
          e~a, e~c, f~b, f~d, e~f
          and e,f are not in {u,v,a,b,c,d}
        """
        nu = adj[u]
        nv = adj[v]
        
        # Find all (a, b) pairs such that a~v, b~u, a~b, and a,b not u,v
        # This forms a square u-v-a-b-u
        pairs = []
        # a must be a neighbor of v (other than u)
        # b must be a neighbor of u (other than v)
        # a~b must hold
        # a,b distinct and not u,v
        
        # Iterate over bits in nv (neighbors of v)
        mask_a = nv & ~(1 << u)
        mask_b = nu & ~(1 << v)
        
        while mask_a:
            a_bit = mask_a & -mask_a
            a = a_bit.bit_length() - 1
            mask_a -= a_bit
            
            # b must be in mask_b, b != a, and a~b
            # candidates for b are neighbors of a in mask_b
            # b must also be a neighbor of u
            common_b = adj[a] & mask_b & ~(1 << a)
            while common_b:
                b_bit = common_b & -common_b
                b = b_bit.bit_length() - 1
                common_b -= b_bit
                pairs.append((a, b))
        
        if len(pairs) < 2:
            return False
        
        # Try all pairs of distinct pairs (a,b) and (c,d)
        # such that {a,b} and {c,d} are disjoint
        for i in range(len(pairs)):
            a, b = pairs[i]
            for j in range(i + 1, len(pairs)):
                c, d = pairs[j]
                
                # Check distinctness
                if a == c or a == d or b == c or b == d:
                    continue
                
                used = {u, v, a, b, c, d}
                used_mask = 0
                for x in used:
                    used_mask |= (1 << x)
                
                # Find e: adjacent to a and c, not in used
                candidates_e = (adj[a] & adj[c]) & ~used_mask
                while candidates_e:
                    e_bit = candidates_e & -candidates_e
                    e = e_bit.bit_length() - 1
                    candidates_e -= e_bit
                    
                    # Find f: adjacent to b and d, not in used, not e, and e~f
                    candidates_f = (adj[b] & adj[d] & adj[e]) & ~used_mask & ~(1 << e)
                    if candidates_f:
                        return True
        
        return False

    # Multi-restart greedy with local search
    num_restarts = 0
    
    # Precompute edge list for random access
    edges_list = all_edges[:]
    
    while time.time() < deadline:
        num_restarts += 1
        
        random.shuffle(edges_list)
        
        # Build adjacency bitmask
        adj = [0] * n
        current_graph = set()
        
        # Greedy phase: try to add each edge in random order
        for edge in edges_list:
            u, v = edge
            # Temporarily add u-v to adj for checking
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if creates_q3(adj, u, v):
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                current_graph.add((u, v))
        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
        
        # Local search: try to add any missing edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for edge in all_edges:
                if edge in current_graph:
                    continue
                u, v = edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not creates_q3(adj, u, v):
                    current_graph.add(edge)
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
        
        # Edge swap local search: remove one edge, try to add two others
        # This helps escape local optima
        if time.time() < deadline:
            # Rebuild adj from current_graph
            adj = [0] * n
            for u, v in current_graph:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Try removing each edge and see if we can add more
            edges_in_graph = list(current_graph)
            random.shuffle(edges_in_graph)
            
            for rem_edge in edges_in_graph:
                if time.time() > deadline:
                    break
                ru, rv = rem_edge
                # Remove
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                current_graph.discard(rem_edge)
                
                # Try to add any missing edge
                added_any = False
                for add_edge in all_edges:
                    if add_edge in current_graph:
                        continue
                    au, av = add_edge
                    adj[au] |= (1 << av)
                    adj[av] |= (1 << au)
                    if not creates_q3(adj, au, av):
                        current_graph.add(add_edge)
                        added_any = True
                    else:
                        adj[au] &= ~(1 << av)
                        adj[av] &= ~(1 << au)
                
                if len(current_graph) > best_edges:
                    best_edges = len(current_graph)
                    best_graph = set(current_graph)
                
                # If we couldn't add more, put the edge back
                if not added_any or len(current_graph) <= best_edges:
                    # Put back
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    current_graph.add(rem_edge)
        
        # Rebuild adj for next iteration
        adj = [0] * n
        for u, v in current_graph:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Safety: limit total restarts to avoid infinite loop
        if num_restarts > 2000:
            break
    
    # Convert to list of lists
    return [[u, v] for u, v in best_graph]
