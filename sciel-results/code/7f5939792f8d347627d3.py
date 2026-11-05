def construct():
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    DEADLINE = 118.0
    n = 16
    total_edges = n * (n - 1) // 2
    
    # Precompute all possible edges as (u, v)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # Map edge to index for fast lookup? Or just use bitmask of edges.
    # Edge bitmask: 120 bits. Python int can handle this.
    # Map (u, v) to bit position
    edge_to_bit = {}
    bit_to_edge = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_to_bit[(u, v)] = idx
        edge_to_bit[(v, u)] = idx
        bit_to_edge[idx] = (u, v)
        
    # Adjacency bitmasks for vertices
    # adj_mask[i] is a bitmask of neighbors of i
    
    def build_adj(edge_mask):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (edge_mask >> edge_to_bit[(i, j)]) & 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def has_q3(adj):
        """
        Check if graph with adjacency masks 'adj' contains a Q3 subgraph.
        Q3 vertices: 0..7
        Q3 edges: 
        (0,1), (0,2), (0,4)
        (1,3), (1,5)
        (2,3), (2,6)
        (3,7)
        (4,5), (4,6)
        (5,7)
        (6,7)
        
        We check for any isomorphic subgraph.
        Since n=16 is small, we can iterate over combinations of 8 vertices? 
        C(16,8) = 12870. For each, check if induced subgraph is Q3? 
        No, "subgraph isomorphic to Q3" means there exists a mapping.
        It does NOT have to be induced.
        So we just need to check if there is a set of 8 vertices and a bijection to Q3 vertices 
        such that all Q3 edges map to edges in G.
        
        A faster way: 
        For each pair of vertices (u, v) that are connected in G, 
        check if they can be part of a Q3.
        In Q3, any edge is part of exactly 2 faces (4-cycles).
        So if (u,v) is in Q3, there must be two distinct 4-cycles containing (u,v) 
        that share the edge (u,v) and have specific structure.
        
        Actually, let's just use the specific structure check.
        Fix u->0, v->1.
        Nbrs of 0 in Q3: 1, 2, 4.
        Nbrs of 1 in Q3: 0, 3, 5.
        So in G, u and v are connected.
        Let A = Nbrs(u) \ {v}
        Let B = Nbrs(v) \ {u}
        We need to pick x, y from A such that x != y. (x->2, y->4)
        We need to pick p, q from B such that p != q. (p->3, q->5)
        Conditions:
        1. (x, p) is an edge in G.
        2. (y, q) is an edge in G.
        3. x, y, p, q, u, v are distinct.
        4. There exists z (->6) such that:
           - z is connected to x and y.
           - z is distinct from u, v, x, y, p, q.
        5. There exists w (->7) such that:
           - w is connected to p, q, z.
           - w is distinct from u, v, x, y, p, q, z.
           
        This is a specific pattern search.
        """
        # Iterate over all edges (u, v) in G
        for u in range(n):
            for v in range(u + 1, n):
                if ((adj[u] >> v) & 1) == 0:
                    continue
                
                # Get neighbors
                # We need to iterate subsets of neighbors
                # To be fast, we can precompute lists
                nbrs_u = [x for x in range(n) if x != v and (adj[u] >> x) & 1]
                nbrs_v = [x for x in range(n) if x != u and (adj[v] >> x) & 1]
                
                # If degree is too low, skip
                if len(nbrs_u) < 2 or len(nbrs_v) < 2:
                    continue
                    
                # Try all pairs (x, y) from nbrs_u
                # x -> 2, y -> 4
                for i_x in range(len(nbrs_u)):
                    for i_y in range(i_x + 1, len(nbrs_u)):
                        x = nbrs_u[i_x]
                        y = nbrs_u[i_y]
                        
                        # x and y must be distinct (guaranteed)
                        # Now find p, q in nbrs_v
                        # p -> 3, q -> 5
                        # Conditions:
                        # (x, p) edge
                        # (y, q) edge
                        # x, y, p, q distinct from u, v and each other
                        
                        # Pre-filter nbrs_v
                        # p must be neighbor of x
                        # q must be neighbor of y
                        # p != q
                        # p, q not in {u, v, x, y}
                        
                        valid_p = []
                        valid_q = []
                        for p in nbrs_v:
                            if p == x or p == y: continue
                            if (adj[x] >> p) & 1:
                                valid_p.append(p)
                        for q in nbrs_v:
                            if q == x or q == y: continue
                            if (adj[y] >> q) & 1:
                                valid_q.append(q)
                                
                        if not valid_p or not valid_q:
                            continue
                            
                        for p in valid_p:
                            for q in valid_q:
                                if p == q: continue
                                
                                used = {u, v, x, y, p, q}
                                
                                # Find z (6): neighbor of x and y
                                z_candidates = []
                                for z in range(n):
                                    if z in used: continue
                                    if (adj[x] >> z) & 1 and (adj[y] >> z) & 1:
                                        z_candidates.append(z)
                                        
                                if not z_candidates:
                                    continue
                                    
                                for z in z_candidates:
                                    used_z = used | {z}
                                    
                                    # Find w (7): neighbor of p, q, z
                                    w_found = False
                                    for w in range(n):
                                        if w in used_z: continue
                                        if (adj[p] >> w) & 1 and (adj[q] >> w) & 1 and (adj[z] >> w) & 1:
                                            w_found = True
                                            break
                                    if w_found:
                                        return True
        return False

    # Initial state: Complete graph is NOT Q3 free.
    # We need a valid starting point.
    # Let's start with a random sparse graph and add edges greedily?
    # Or start with K_{8,8} and remove edges?
    
    # Better: Start with a known Q3-free graph.
    # Random graph G(16, 0.5) is likely Q3-free?
    # Let's try to construct a valid graph by random edge addition.
    
    def random_valid_graph():
        edge_mask = 0
        adj = [0] * n
        # Try to add edges randomly until no more can be added (greedy)
        # Shuffle edges
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        for e in shuffled:
            u, v = e
            # Add edge
            bit = edge_to_bit[e]
            edge_mask |= (1 << bit)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if Q3 exists
            if has_q3(adj):
                # Remove edge
                edge_mask &= ~(1 << bit)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return edge_mask, adj

    # Initialize
    current_mask, current_adj = random_valid_graph()
    current_score = bin(current_mask).count("1")
    best_mask = current_mask
    best_score = current_score
    best_adj = current_adj
    
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < DEADLINE:
        # Generate a move
        # 1. Add a random missing edge
        # 2. Remove a random present edge
        # 3. Swap: remove one, add one
        
        move_type = random.random()
        
        new_mask = current_mask
        new_adj = current_adj[:] # Copy is expensive, let's use deltas or rebuild if needed
        # For efficiency, we will modify in place and revert if rejected
        
        if move_type < 0.5:
            # Add edge
            # Find a missing edge
            # To avoid iterating 120 edges, pick random index
            for _ in range(10):
                idx = random.randint(0, total_edges - 1)
                if (current_mask >> idx) & 1 == 0:
                    u, v = bit_to_edge[idx]
                    # Apply
                    current_mask |= (1 << idx)
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    break
            else:
                continue # No missing edges found (complete graph)
                
        elif move_type < 0.8:
            # Remove edge
            # Find a present edge
            present_indices = [i for i in range(total_edges) if (current_mask >> i) & 1]
            if not present_indices:
                continue
            idx = random.choice(present_indices)
            u, v = bit_to_edge[idx]
            current_mask &= ~(1 << idx)
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
        else:
            # Swap
            # Pick present idx1, missing idx2
            present_indices = [i for i in range(total_edges) if (current_mask >> i) & 1]
            missing_indices = [i for i in range(total_edges) if (current_mask >> i) & 1 == 0]
            if not present_indices or not missing_indices:
                continue
            idx1 = random.choice(present_indices)
            idx2 = random.choice(missing_indices)
            u1, v1 = bit_to_edge[idx1]
            u2, v2 = bit_to_edge[idx2]
            
            # Remove idx1
            current_mask &= ~(1 << idx1)
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            
            # Add idx2
            current_mask |= (1 << idx2)
            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            
        # Check validity
        if has_q3(current_adj):
            # Revert
            if move_type < 0.5:
                idx = idx # Wait, idx is not defined in scope if loop didn't break? 
                # The logic above is slightly messy. Let's refactor the move generation to be cleaner.
                pass # This block is unreachable if we structure it better.
            
            # Actually, let's just revert based on what we did.
            # This is getting complicated with in-place modification.
            # Let's use a "try" block with explicit revert.
            
            # Re-implementing move logic for safety
            pass # See below
            
        else:
            new_score = bin(current_mask).count("1")
            delta_score = new_score - current_score
            
            # Acceptance
            if delta_score > 0:
                # Always accept improvement
                pass
            else:
                # Accept worse with probability exp(delta/T)
                if random.random() < np.exp(delta_score / T):
                    pass
                else:
                    # Reject and revert
                    # We need to know what we did to revert.
                    # This is why I need to store the move.
                    pass
                    
        # This in-place approach is error-prone. 
        # Let's switch to a cleaner "Candidate" approach.
        
        # Break to rewrite the loop properly in the final code.
        break

    # The logic above is a sketch. I will write the full clean version in the replacement.
    return [[u, v] for u, v in all_edges if (best_mask >> edge_to_bit[(u, v)]) & 1]
