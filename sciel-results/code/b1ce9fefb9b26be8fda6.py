import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Try to find a graph for n=32 down to 29
    for n in range(32, 27, -1):
        if time.time() > deadline:
            break
        result = try_construct(n, deadline)
        if result:
            return result
            
    # Fallback to seed solution
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}

def try_construct(n, deadline):
    # Bitset representation
    # adj[i] is an int where bit j is 1 if edge (i,j) exists
    
    def alpha_check(adj):
        # Greedy maximal independent set
        # Returns size of the largest found
        unvisited = (1 << n) - 1
        max_alpha = 0
        # Simple greedy: pick vertex with min degree, remove it and its neighbors
        # This finds a maximal independent set, not necessarily maximum.
        # For checking alpha >= 9, a greedy heuristic is usually sufficient to find one if it exists,
        # or we can use a few random restarts of the greedy.
        
        # Better: Randomized Greedy
        for _ in range(50):
            visited = 0
            current_set = 0
            remaining = (1 << n) - 1
            while remaining:
                # Pick random vertex from remaining
                # To speed up, we can just pick the lowest set bit
                lsb = remaining & -remaining
                v = lsb.bit_length() - 1
                # Add v to independent set
                current_set |= (1 << v)
                # Remove v and its neighbors from remaining
                neighbors = adj[v]
                remaining &= ~(neighbors | (1 << v))
            
            size = bin(current_set).count('1')
            if size > max_alpha:
                max_alpha = size
            if max_alpha >= 9:
                return max_alpha, current_set
        return max_alpha, 0

    def find_c5_edge(adj):
        # Returns an edge (u, v) that is part of a C5, or None
        # Check for C5 by iterating non-edges (u, v)
        # and looking for a path of length 4 between them.
        # Path: u - a - b - c - v
        # a in N(u), c in N(v), b in N(a) \cap N(c)
        # Vertices u, v, a, b, c must be distinct.
        # Also, no chords: 
        # u-c not edge, u-b not edge, v-a not edge, v-b not edge, a-c not edge? 
        # Actually, if there are chords, it's not a simple C5 subgraph, but the task says "no 5-cycle as a subgraph".
        # A subgraph C5 means we need 5 vertices forming a cycle. Chords don't matter for the existence of the cycle, 
        # but they might form a smaller cycle. 
        # "No 5-cycle as a subgraph" usually means no induced C5? 
        # "Containing no 5-cycle (as a subgraph)" means the graph does not contain a subgraph isomorphic to C5.
        # If there are chords, the subgraph might still contain a C5 if the chords don't break the cycle?
        # No, if there is a chord, the cycle is not a simple cycle of length 5 in the subgraph induced by those 5 vertices?
        # Actually, a subgraph isomorphic to C5 requires 5 vertices and 5 edges forming a cycle.
        # If there are extra edges, the subgraph *contains* the C5.
        # So we just need to find any 5 vertices that form a cycle of length 5.
        
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue # Must be non-adjacent for the "endpoints" of the path of length 4
                
                # Find a in N(u)
                a_mask = adj[u] & ~(1 << v) # a != v
                # Find c in N(v)
                c_mask = adj[v] & ~(1 << u) # c != u
                
                # We need a, c distinct, a != c
                # And b in N(a) \cap N(c)
                # b distinct from u, v, a, c
                
                # Iterate a
                a_list = []
                mask = a_mask
                while mask:
                    lsb = mask & -mask
                    a = lsb.bit_length() - 1
                    a_list.append(a)
                    mask ^= lsb
                
                for a in a_list:
                    # b must be in N(a) \cap N(c)
                    # But c is not fixed yet.
                    # This approach is O(N^3) or worse.
                    pass
        
        # Faster C5 check:
        # Iterate over all pairs of vertices (u, v) that are NOT adjacent.
        # Let S = N(u) \cap N(v).
        # For each pair x, y in S:
        #   Check if x and y have a common neighbor z such that z is not u, v, x, y.
        #   And check that the 5 vertices u, v, x, y, z form a simple cycle?
        #   The cycle would be u-x-z-y-v-u.
        #   We need to ensure no chords that make it not a C5? 
        #   Actually, if u-x-z-y-v-u is a path, and u-v is not an edge, it's a C5.
        #   Chords like u-z would create a triangle u-x-z-u, but the C5 subgraph still exists (using edges u-x, x-z, z-y, y-v, v-u? No, v-u is not an edge).
        #   The cycle is u-x-z-y-v-... wait.
        #   Edges: (u,x), (x,z), (z,y), (y,v).
        #   We need (v,u) to close the cycle? No, (v,u) is NOT an edge.
        #   So u-x-z-y-v is a path of length 4.
        #   To form a cycle, we need an edge between u and v? No, that's the non-edge.
        #   A 5-cycle is a cycle of length 5.
        #   If u and v are not adjacent, a path of length 4 between them does NOT form a cycle.
        #   It forms a path.
        #   To form a C5, we need a path of length 4 between u and v, AND u and v must be connected? No.
        #   A cycle of length 5 has 5 edges.
        #   If u and v are not adjacent, they cannot be part of the same edge in the cycle.
        #   So the cycle must go u-a-b-c-d-v-... no, it must return to u.
        #   So the cycle is u-a-b-c-d-u.
        #   Then u and d are adjacent, u and a are adjacent.
        #   So u and v (if v=d) are adjacent.
        #   So in a C5, every pair of vertices in the cycle is either adjacent or distance 2.
        #   No pair is non-adjacent? 
        #   In C5, vertices at distance 2 are non-adjacent.
        #   So pick a non-edge (u, v) in the C5. They are distance 2.
        #   Path u-x-v of length 2.
        #   The rest of the cycle is u-y-z-v of length 3.
        #   So we need to find x, y, z such that:
        #   1. u-x, x-v are edges.
        #   2. u-y, y-z, z-v are edges.
        #   3. x, y, z are distinct and distinct from u, v.
        #   4. x is not adjacent to y, z? (Chords)
        #   If chords exist, it's still a C5 subgraph? 
        #   Yes, a subgraph isomorphic to C5 exists if we can pick 5 edges forming a cycle.
        #   So we just need to find the path structure.
        
        # Algorithm:
        # For each non-edge (u, v):
        #   Let X = N(u) \cap N(v). (Common neighbors)
        #   For each x in X:
        #      We need a path of length 3 from u to v that does not use x.
        #      Path: u - a - b - v.
        #      a in N(u) \ {x, v}, b in N(v) \ {x, u}.
        #      a-b is an edge.
        #      a, b distinct from u, v, x.
        #      a != b.
        
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                
                # Common neighbors
                common = adj[u] & adj[v]
                # Iterate x in common
                x_mask = common
                while x_mask:
                    lsb = x_mask & -x_mask
                    x = lsb.bit_length() - 1
                    x_mask ^= lsb
                    
                    # Find a in N(u) \ {x, v}
                    # Find b in N(v) \ {x, u}
                    # Such that a-b is an edge.
                    
                    # a_mask = adj[u] & ~(1 << x) & ~(1 << v)
                    # b_mask = adj[v] & ~(1 << x) & ~(1 << u)
                    # We need a in a_mask, b in b_mask, and (adj[a] >> b) & 1.
                    
                    a_mask = adj[u] & ~(1 << x) & ~(1 << v)
                    b_mask = adj[v] & ~(1 << x) & ~(1 << u)
                    
                    # Iterate a
                    a_temp = a_mask
                    while a_temp:
                        lsb_a = a_temp & -a_temp
                        a = lsb_a.bit_length() - 1
                        a_temp ^= lsb_a
                        
                        # Check if a is connected to any b in b_mask
                        # b_candidates = b_mask & adj[a]
                        if b_mask & adj[a]:
                            # Found a path u-a-b-v and u-x-v
                            # Vertices: u, v, x, a, b
                            # Edges: (u,x), (x,v), (u,a), (a,b), (b,v)
                            # This forms a cycle u-x-v-b-a-u of length 5?
                            # Yes, if all vertices are distinct.
                            # u != v (non-edge)
                            # x != u, v (x in N(u) \cap N(v))
                            # a != u, v, x (a in N(u) \ {x, v})
                            # b != u, v, x (b in N(v) \ {x, u})
                            # a != b? 
                            # If a == b, then u-a-v is a triangle, and u-x-v is a triangle.
                            # Cycle u-x-v-a-u is length 4. Not a C5.
                            # So we must ensure a != b.
                            # Also, we should check if this specific set of edges forms a C5.
                            # The edges are (u,x), (x,v), (v,b), (b,a), (a,u).
                            # This is a cycle of length 5.
                            # Is it a simple cycle? Yes, if vertices are distinct.
                            # Are they distinct?
                            # u, v distinct.
                            # x distinct from u, v.
                            # a distinct from u, v, x.
                            # b distinct from u, v, x.
                            # a distinct from b?
                            # If a == b, then (u,a) and (a,v) are edges, and (u,x) and (x,v) are edges.
                            # The cycle would be u-x-v-a-u (length 4).
                            # So we must check a != b.
                            
                            # Find the specific b
                            b_mask_specific = b_mask & adj[a]
                            if b_mask_specific:
                                lsb_b = b_mask_specific & -b_mask_specific
                                b = lsb_b.bit_length() - 1
                                if a != b:
                                    return (a, b) # Return an edge from the C5 to remove
        return None

    # Main repair loop
    for restart in range(2000):
        if time.time() > deadline:
            return None
        
        # Initialize random graph
        adj = [0] * n
        # p = 0.5 is a good start
        p = 0.5
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        for step in range(1000):
            if time.time() > deadline:
                return None
            
            # Check Alpha
            alpha_size, indep_set = alpha_check(adj)
            if alpha_size >= 9:
                # Add an edge in the independent set to break it
                # Get vertices in indep_set
                verts = []
                mask = indep_set
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    mask ^= lsb
                
                # Pick random pair
                if len(verts) >= 2:
                    u, v = random.sample(verts, 2)
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    continue
            
            # Check C5
            edge_to_remove = find_c5_edge(adj)
            if edge_to_remove:
                u, v = edge_to_remove
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                continue
            
            # Both constraints satisfied
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return None
