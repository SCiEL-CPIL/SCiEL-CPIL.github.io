"""Strong algorithm for C4-free edge set in Q7 using max independent set."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Enumerate all edges of Q7
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
    
    m = len(edges)  # 448
    
    # Build conflict graph: two edges conflict if they form a 4-cycle
    # Edge e1 = (x, x^bit_i) and e2 = (y, y^bit_j) conflict if:
    #   x ^ y = bit_i ^ bit_j (i.e., x and y differ in exactly bits i and j)
    #   and the four vertices form a cycle
    
    # For each edge, find conflicting edges
    # Edge idx corresponds to (x, y) where y = x ^ (1<<i)
    # Conflicts with edges (x^(1<<j), y^(1<<j)) for j != i
    # That's 6 conflicts per edge
    
    conflict_mask = [0] * m
    
    for idx, (x, y) in enumerate(edges):
        # Find which bit differs
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            # The conflicting edge is (x ^ (1<<j), y ^ (1<<j))
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            c_idx = edge_to_idx[(min(c, e), max(c, e))]
            conflict_mask[idx] |= (1 << c_idx)
    
    # Now we want maximum independent set in conflict graph
    # = maximum clique in complement graph
    # Complement adjacency: adj[k] = bitmask of edges NOT conflicting with k (and not k itself)
    
    all_mask = (1 << m) - 1
    
    # comp_adj[k]: edges that can be in a clique with k (i.e., don't conflict with k)
    # In the complement of conflict graph, two vertices are adjacent if they DON'T conflict
    # So comp_adj[k] = all_mask ^ conflict_mask[k] ^ (1 << k)
    
    comp_adj = [all_mask ^ conflict_mask[k] ^ (1 << k) for k in range(m)]
    
    # Maximum clique algorithm using Tomita's approach with bit-parallelism
    # We use a recursive branch-and-bound with greedy coloring
    
    best_clique = []
    best_size = 0
    
    # Use numpy for bit operations where helpful, but Python ints work well for bitsets up to 448 bits
    
    def popcount(x):
        return bin(x).count('1')
    
    def greedy_color(P):
        """Greedy coloring of subgraph induced by P.
        Returns: list of color classes (bitmasks), where each color class is an independent set in comp_adj
                 (i.e., a clique in conflict graph)."""
        color_classes = []
        remaining = P
        while remaining:
            # Pick a vertex, assign it a new color class
            lowbit = remaining & (-remaining)
            v = lowbit.bit_length() - 1
            
            # Current color class
            cc = 0
            avail = remaining
            while avail:
                lowbit = avail & (-avail)
                v = lowbit.bit_length() - 1
                cc |= lowbit
                # Remove v and its neighbors from avail
                avail &= ~(comp_adj[v] | lowbit)
            
            color_classes.append(cc)
            remaining &= ~cc
        
        return color_classes
    
    def max_clique_rec(P, clique, depth):
        """Recursive maximum clique search.
        P: bitmask of candidates
        clique: list of vertices in current clique
        depth: current clique size
        """
        nonlocal best_clique, best_size
        
        if depth + popcount(P) <= best_size:
            return
        
        if P == 0:
            if depth > best_size:
                best_size = depth
                best_clique = clique[:]
            return
        
        # Greedy coloring for upper bound
        color_classes = greedy_color(P)
        max_color = len(color_classes)
        
        if depth + max_color <= best_size:
            return
        
        # Branch: process vertices in order of color (reverse color order for better pruning)
        # For each color class from highest to lowest, try adding vertices from that class
        for c in range(max_color - 1, -1, -1):
            if depth + c + 1 <= best_size:
                break
            # Try vertices in color class c
            # Process in some order
            cc = color_classes[c]
            # Get list of vertices in this color class
            # We'll iterate through them
            # For each vertex v in cc, branch on including v
            # After including v, new P = P & comp_adj[v] & (vertices with color > c)
            
            # Get list of vertices in cc
            verts_in_cc = []
            tmp = cc
            while tmp:
                lowbit = tmp & (-tmp)
                v = lowbit.bit_length() - 1
                verts_in_cc.append(v)
                tmp ^= lowbit
            
            for v in verts_in_cc:
                if depth + 1 + c <= best_size:
                    break
                # Include v in clique
                new_P = P & comp_adj[v]
                # Only keep vertices that come after v in our ordering
                # For simplicity, just use new_P and the coloring will handle bounds
                max_clique_rec(new_P, clique + [v], depth + 1)
                # Remove v from P for subsequent iterations
                P ^= (1 << v)
    
    # Start with all vertices
    start_P = all_mask
    max_clique_rec(start_P, [], 0)
    
    # best_clique is a set of edge indices that form a clique in complement = independent set in conflict graph
    # This means no two edges in best_clique conflict, i.e., no 4-cycle is formed
    result = []
    for idx in best_clique:
        x, y = edges[idx]
        result.append([x, y])
    
    return result

# EVOLVE-BLOCK-END
