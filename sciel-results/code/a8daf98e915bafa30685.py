import random
import time
import math

def construct():
    n = 20
    
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    m = len(all_edges)

    def is_c4_free(adj, u, v):
        """Check if adding (u,v) to graph with adjacencies 'adj' creates a C4."""
        # A C4 is formed if there are two vertices x, y such that
        # x is connected to u and v, and y is connected to u and v? No.
        # C4: u-x-v-y-u. 
        # Adding (u,v) creates a C4 if there exist x, y in N(u) \cap N(v) ? No.
        # If we add (u,v), a C4 involving (u,v) must be of the form u-v-x-y-u.
        # This requires x in N(v) \ N(u) and y in N(u) \ N(v) such that (x,y) is an edge?
        # No, simpler: 
        # The new edge (u,v) completes a C4 if there is a path of length 3 between u and v that does not use (u,v).
        # Path u-x-y-v. 
        # So x must be in N(u) \ N(v) and y must be in N(v) \ N(u) and (x,y) must be an edge?
        # Wait, x is neighbor of u. y is neighbor of v. x-y is an edge.
        # Also x cannot be v (since v not in N(u) yet) and y cannot be u.
        # So: For all x in N(u), for all y in N(v), if (x,y) is an edge and x!=v and y!=u, 
        # then adding (u,v) creates a cycle u-x-y-v-u. 
        # Is this a C4? 
        # If x==y, then u-x-v-u is a triangle (C3), not C4. But we are checking C4-free. 
        # Triangles are allowed in C4-free graphs? Yes. 
        # But if x==y, we have a triangle. Does adding (u,v) create a C4? 
        # No, it creates a triangle. 
        # So we only care if there is a path of length 3.
        # Path u-x-y-v. x != y.
        # So: 
        # N_u = neighbors of u
        # N_v = neighbors of v
        # For each x in N_u:
        #   For each y in N_v:
        #     if x == y: continue (forms triangle, not C4)
        #     if x == v or y == u: continue (should not happen if u,v not connected)
        #     if (x,y) is an edge: return False (creates C4)
        
        # Optimization:
        # We can check: 
        # Let S = N(u) & N(v). These are common neighbors. 
        # If we add (u,v), any common neighbor x forms a triangle u-v-x-u.
        # To form a C4, we need a "cross" connection.
        # Actually, the standard check for C4-free addition:
        # Adding (u,v) creates a C4 if and only if there exist x, y such that 
        # x in N(u), y in N(v), x != y, and (x,y) in E.
        # 
        # Let's implement this efficiently.
        nu = adj[u]
        nv = adj[v]
        
        # Iterate over neighbors of u
        temp_u = nu
        while temp_u:
            lsb = temp_u & -temp_u
            x = lsb.bit_length() - 1
            temp_u ^= lsb
            
            if x == v: continue
            
            # Check if x is connected to any neighbor of v (other than u)
            # We need to check edges from x to N(v).
            # If x is in N(v), it's a common neighbor (triangle case).
            # If x is not in N(v), we check if x has any neighbor in N(v).
            
            # Actually, we can just check:
            # Does x have any neighbor in N(v) \ {u, x}?
            # Note: u is not in N(v) yet.
            
            # Let's just iterate N(v)
            # This is O(deg(u) * deg(v))
            # For n=20, deg is small.
            
            temp_v = nv
            while temp_v:
                lsb_v = temp_v & -temp_v
                y = lsb_v.bit_length() - 1
                temp_v ^= lsb_v
                
                if y == u: continue
                if x == y: continue # Triangle
                
                # Check if (x,y) is an edge
                if (adj[x] >> y) & 1:
                    return False
        
        return True

    def edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def add_edge(adj, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def build_random_c4_free_graph():
        adj = [0] * n
        edges = list(all_edges)
        random.shuffle(edges)
        for u, v in edges:
            if is_c4_free(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Initialize G1 and G2
    # Strategy: Build G1, then build G2 avoiding edges in G1
    A1 = build_random_c4_free_graph()
    
    # Build G2: Start with empty, add edges not in A1 that keep G2 C4-free
    A2 = [0] * n
    edges_for_g2 = [e for e in all_edges if not ((A1[e[0]] >> e[1]) & 1)]
    random.shuffle(edges_for_g2)
    for u, v in edges_for_g2:
        if is_c4_free(A2, u, v):
            add_edge(A2, u, v)
            
    best_A1 = [x for x in A1]
    best_A2 = [x for x in A2]
    best_edges = edge_count(best_A1) + edge_count(best_A2)
    
    curr_A1 = [x for x in A1]
    curr_A2 = [x for x in A2]
    curr_edges = edge_count(curr_A1) + edge_count(curr_A2)
    
    T = 1.0
    cooling = 0.999
    
    # Simulated Annealing
    steps = 0
    max_steps = 100000
    
    while steps < max_steps and time.time() < deadline:
        steps += 1
        
        if time.time() > deadline - 5:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an edge to G1
            u, v = random.choice(all_edges)
            if not ((curr_A1[u] >> v) & 1) and not ((curr_A2[u] >> v) & 1):
                if is_c4_free(curr_A1, u, v):
                    add_edge(curr_A1, u, v)
                    curr_edges += 1
                    if curr_edges > best_edges:
                        best_edges = curr_edges
                        best_A1 = [x for x in curr_A1]
                        best_A2 = [x for x in curr_A2]
                    T *= 0.99
                else:
                    T *= 0.95
            else:
                T *= 0.99
        elif move_type < 0.75:
            # Try to add an edge to G2
            u, v = random.choice(all_edges)
            if not ((curr_A1[u] >> v) & 1) and not ((curr_A2[u] >> v) & 1):
                if is_c4_free(curr_A2, u, v):
                    add_edge(curr_A2, u, v)
                    curr_edges += 1
                    if curr_edges > best_edges:
                        best_edges = curr_edges
                        best_A1 = [x for x in curr_A1]
                        best_A2 = [x for x in curr_A2]
                    T *= 0.99
                else:
                    T *= 0.95
            else:
                T *= 0.99
        else:
            # Try to remove an edge from G1 or G2 to allow better swaps later?
            # Or just random removal to escape local minima
            if random.random() < 0.5:
                # Remove from G1
                # Pick a random edge in G1
                # Find an edge
                found = False
                for _ in range(10): # Try a few times
                    u, v = random.choice(all_edges)
                    if (curr_A1[u] >> v) & 1:
                        remove_edge(curr_A1, u, v)
                        curr_edges -= 1
                        found = True
                        break
                if not found:
                    T *= 0.99
            else:
                # Remove from G2
                found = False
                for _ in range(10):
                    u, v = random.choice(all_edges)
                    if (curr_A2[u] >> v) & 1:
                        remove_edge(curr_A2, u, v)
                        curr_edges -= 1
                        found = True
                        break
                if not found:
                    T *= 0.99
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final polish: Greedy add
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if not ((best_A1[u] >> v) & 1) and not ((best_A2[u] >> v) & 1):
            if is_c4_free(best_A1, u, v):
                add_edge(best_A1, u, v)
                best_edges += 1
            elif is_c4_free(best_A2, u, v):
                add_edge(best_A2, u, v)
                best_edges += 1

    # Construct result
    result = []
    for u, v in all_edges:
        if (best_A1[u] >> v) & 1 or (best_A2[u] >> v) & 1:
            result.append([u, v])
            
    return result
