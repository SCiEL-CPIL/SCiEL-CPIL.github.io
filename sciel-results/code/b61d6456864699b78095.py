import time
import sys

def construct():
    start_time = time.time()
    
    # We try n=18 first as per directive, then fallback to 17, 16...
    # If n=18 is not found quickly, we switch to n=17 which is known to exist.
    for n in [18, 17, 16]:
        if time.time() - start_time > 115:
            break
            
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * num_edges
        
        # Symmetry breaking:
        # Fix colors of edges (0,1), (0,2), (0,3) to 0, 1, 2 respectively
        # This breaks color permutation symmetry
        # Index 0: (0,1)
        # Index 1: (0,2)
        # Index 2: (0,3)
        
        def set_edge(u, v, c, idx):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            colors[idx] = c

        def unset_edge(u, v, c, idx):
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1

        # Apply symmetry breaking
        if n >= 4:
            set_edge(0, 1, 0, 0)
            set_edge(0, 2, 1, 1)
            set_edge(0, 3, 2, 2)
            start_idx = 3
        else:
            start_idx = 0

        def has_path_len_5(u, v, c):
            # Check if there is a simple path of length 5 from u to v in graph c
            # This creates a C6 if we add edge (u,v)
            # Path: u -> n1 -> n2 -> n3 -> n4 -> v
            # We can do this with a fast bitmask DFS
            
            # Optimization: If u and v are not connected in a way that allows length 5, return False quickly?
            # Simple DFS with backtracking using bitmasks is standard.
            
            # We use a recursive helper to avoid stack overhead in the main loop?
            # Python recursion is slow. Iterative is better.
            
            # Let's use a stack of (node, depth, visited_mask)
            # But we can prune: if depth + (dist_to_v_lower_bound) > 5, skip?
            # Hard to compute dist quickly.
            
            # Alternative:
            # Expand from u for 2 steps, from v for 2 steps, check intersection?
            # Path u-a-b-c-d-v.
            # If we take 2 steps from u: set U2.
            # If we take 2 steps from v: set V2.
            # If U2 and V2 intersect at some node x, we have path u-a-x-b-v? No, that's length 4.
            # We need length 5.
            # u-a-b-c-d-v.
            # Middle node is c.
            # Path from u to c of length 2: u-a-c.
            # Path from v to c of length 2: v-b-c.
            # If there exists a node c such that dist(u,c)=2 and dist(v,c)=2 in G_c, and the paths are vertex disjoint (except c)?
            # Actually, if dist(u,c)=2 and dist(v,c)=2, we have u-a-c and v-b-c.
            # If a != b and a != v and b != u, then u-a-c-b-v is a path of length 4? No, u-a-c-b-v is length 4 edges.
            # We need length 5 edges.
            
            # Let's stick to the DFS but optimize it.
            
            # We can check:
            # Is there a node x such that there is a path of length 2 from u to x and path of length 3 from x to v?
            # Or path of length 3 from u to x and path of length 2 from x to v?
            
            # Let's implement a fast iterative DFS.
            # To make it faster, we can limit the branching.
            
            stack = [(u, 0, 1 << u)]
            while stack:
                node, depth, visited = stack.pop()
                if depth == 5:
                    if node == v:
                        return True
                    continue
                
                neighbors = adj[c][node]
                # Iterate through set bits
                while neighbors:
                    lsb = neighbors & -neighbors
                    nxt = lsb.bit_length() - 1
                    neighbors ^= lsb
                    
                    if visited & lsb:
                        continue
                    
                    if depth == 4 and nxt == v:
                        return True
                        
                    stack.append((nxt, depth + 1, visited | lsb))
            return False

        def backtrack(idx):
            if time.time() - start_time > 110:
                return False
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            
            for c in range(4):
                if has_path_len_5(u, v, c):
                    continue
                
                set_edge(u, v, c, idx)
                
                if backtrack(idx + 1):
                    return True
                
                unset_edge(u, v, c, idx)
            
            return False

        if backtrack(start_idx):
            return {'n': n, 'colors': colors}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
