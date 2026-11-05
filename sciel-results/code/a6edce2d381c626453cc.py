import time
import random
import numpy as np

def construct():
    n = 40
    deadline = 115.0
    start_time = time.time()
    
    # Bitset representation of adjacency
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge
    adj = [0] * n
    
    # A2[i] is an integer where bit j is 1 if there is a path of length 2 from i to j
    # A2[i] = union of adj[k] for all k in adj[i]
    A2 = [0] * n
    
    def update_A2():
        for i in range(n):
            mask = 0
            # Iterate over set bits in adj[i]
            a = adj[i]
            while a:
                lsb = a & -a
                k = lsb.bit_length() - 1
                mask |= adj[k]
                a ^= lsb
            A2[i] = mask

    def check_c6_add(u, v):
        # Returns True if adding (u,v) creates a C6
        # C6: u-x-y-z-w-v
        # x in N(u), w in N(v)
        # Path x-y-z-w of length 3 exists iff N(x) & A2[w] != 0
        # Note: We must ensure the path doesn't reuse vertices in a way that creates a smaller cycle?
        # The problem asks for "no six-cycle". Usually this means no induced C6 or no C6 as a subgraph?
        # "containing no six-cycle" usually means C6 is not a subgraph.
        # If we add (u,v), and there is a path of length 5, it forms a C6.
        # The path u-x-y-z-w-v must be simple for it to be a C6?
        # If the path repeats vertices, it might form a smaller cycle or a non-simple walk.
        # However, if a path of length 5 exists, it implies a C6 subgraph exists.
        # The check N(x) & A2[w] != 0 finds a walk of length 3.
        # Does a walk of length 3 imply a simple path of length 3?
        # If x-y-z-w is a walk, and vertices repeat, e.g., x-y-x-w, then x and w are distance 2 apart?
        # If x-y-x-w, then adj[x] has y, adj[y] has x, adj[x] has w.
        # Then N(x) & A2[w]: y is in N(x). Is y in A2[w]?
        # A2[w] contains neighbors of neighbors of w. Neighbors of w include x (since x-w edge? No, x-y-x-w implies x-w is NOT necessarily an edge, wait.
        # Path x-y-x-w means edges (x,y), (y,x), (x,w).
        # So x is neighbor of w.
        # A2[w] contains neighbors of x. So y is in A2[w].
        # So N(x) & A2[w] is non-empty (contains y).
        # But the path u-x-y-x-w-v uses x twice. This is not a simple cycle.
        # Does it contain a C6?
        # If u-x-y-x-w-v is a walk, and x is repeated, the cycle formed is u-x-y-x-w-v-u?
        # Length 6. But x is repeated. So it's not a simple cycle.
        # However, if x is repeated, we have a triangle x-y-x? No, simple graph.
        # The walk x-y-x implies edge (x,y).
        # The walk x-w implies edge (x,w).
        # So we have triangle u-x-w-v? No.
        # If the walk is not simple, it might not form a C6.
        # But for n=40 and dense graphs, simple paths are likely.
        # To be safe, we should check for simple paths? That's expensive.
        # However, standard C6-free checks often use the matrix power method which counts walks.
        # If (A^5)[u,v] > 0, it counts walks.
        # If there is a non-simple walk of length 5, does it imply a C6?
        # Not necessarily. But in practice, for this density, it's a good proxy.
        # Let's assume the standard definition where subgraph C6 is forbidden.
        # If a non-simple walk exists, it might mask the absence of a simple C6.
        # But if a simple C6 exists, the walk check will catch it.
        # So false positives (rejecting valid additions) are possible, false negatives (accepting invalid) are not.
        # This is safe for validity, but might limit edge count.
        
        # Optimization: We can prune.
        # x in N(u), w in N(v).
        # We need N(x) & A2[w] != 0.
        # Note: A2[w] includes w's neighbors' neighbors.
        # If x == w, we check N(x) & A2[x]. This is triangles involving x?
        # If x == w, path is u-x-y-x-v? No, x-y-z-w with x=w => x-y-z-x.
        # This is a cycle of length 3 (triangle) if y-z is edge?
        # If x-y-z-x is a triangle, then u-x-z-v?
        # This is getting complicated. Let's just do the bit check. It's fast enough.
        
        a_u = adj[u]
        a_v = adj[v]
        
        # Iterate neighbors of u
        x_mask = a_u
        while x_mask:
            lsb_x = x_mask & -x_mask
            x = lsb_x.bit_length() - 1
            x_mask ^= lsb_x
            
            # Iterate neighbors of v
            w_mask = a_v
            while w_mask:
                lsb_w = w_mask & -w_mask
                w = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                
                # Check if there is a path of length 3 between x and w
                # This is equivalent to N(x) & A2[w] != 0
                if adj[x] & A2[w]:
                    return True
        
        return False

    # Initialize with K5s to have a decent start
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    update_A2()
    
    best_edges = sum(bin(a).count("1") for a in adj) // 2
    best_adj = adj[:]
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < 0.1:
            T = 10.0 # Restart annealing
            # Maybe randomize a bit?
            # Let's just continue, SA with reset T is a form of Iterated Local Search
            
        # Pick a random move
        # 70% chance: Add edge
        # 30% chance: Swap (Remove one, Add one)
        
        if random.random() < 0.7:
            # Add edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj[u] >> v) & 1:
                continue
            
            # Check validity
            if not check_c6_add(u, v):
                # Valid
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                update_A2()
                current_edges = sum(bin(a).count("1") for a in adj) // 2
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Invalid, skip
                pass
        else:
            # Swap: Remove (a,b), Add (c,d)
            # Pick random edge to remove
            edges_list = []
            for i in range(n):
                a = adj[i]
                while a:
                    lsb = a & -a
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges_list.append((i, j))
                    a ^= lsb
            
            if not edges_list:
                continue
                
            a, b = random.choice(edges_list)
            
            # Save state
            adj_backup = adj[:]
            A2_backup = A2[:]
            edges_backup = sum(bin(x).count("1") for x in adj) // 2
            
            # Remove (a,b)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            update_A2()
            
            # Try to add a random edge (c,d)
            c = random.randint(0, n - 1)
            d = random.randint(0, n - 1)
            if c == d or (adj[c] >> d) & 1:
                # Revert
                adj = adj_backup
                A2 = A2_backup
                continue
                
            if not check_c6_add(c, d):
                adj[c] |= (1 << d)
                adj[d] |= (1 << c)
                update_A2()
                new_edges = sum(bin(x).count("1") for x in adj) // 2
                
                # Accept if better or equal, or with probability exp(-(delta)/T) if worse
                # Since we removed 1 and added 1, delta is usually 0.
                # But we might have removed an edge that was "blocking" more edges?
                # Here we just swap. Edge count stays same.
                # So we accept if it allows future improvements?
                # For simplicity, accept if edge count is same (neutral move) with high prob.
                # Or just accept if it doesn't decrease edge count.
                if new_edges >= edges_backup:
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = adj[:]
                else:
                    # Revert
                    adj = adj_backup
                    A2 = A2_backup
            else:
                # Revert
                adj = adj_backup
                A2 = A2_backup
        
        T *= cooling

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        a = best_adj[u]
        while a:
            lsb = a & -a
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            a ^= lsb
            
    return edges
