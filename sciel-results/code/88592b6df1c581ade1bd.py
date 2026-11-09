"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Keep 10s margin

    # Helper to convert edges to adjacency matrix
    def edges_to_adj(n, edges):
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj

    def adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges

    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        adj = np.zeros((q, q), dtype=np.int8)
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    def count_bad_sets(n, adj):
        """Count K5 and independent 5-sets."""
        k5_count = 0
        # Precompute common neighbors for speed? 
        # For n=38, O(n^5) is ~790M ops. Too slow in pure Python loops if done naively.
        # We need a faster check.
        
        # Check K5:
        # For each pair (u,v), if adj[u,v]==1, count K3 in common neighbors.
        # K3 count in a set S is O(|S|^3).
        # This is still slow.
        
        # Optimization: Use bitmasks? 
        # n=38 fits in a 64-bit int.
        # adj_mask[i] is an int with bits set for neighbors of i.
        
        masks = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if adj[i, j]:
                    m |= (1 << j)
            masks[i] = m
            
        def count_k3_in_mask(m):
            # Count triangles in the subgraph induced by vertices in mask m
            # Iterate over bits in m
            count = 0
            # Extract list of nodes
            nodes = []
            mm = m
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                nodes.append(idx)
                mm ^= lsb
                
            # Count triangles in nodes
            ln = len(nodes)
            if ln < 3: return 0
            
            # For each pair in nodes, check if they share a common neighbor in nodes
            # Or just iterate triples? O(ln^3). ln is small.
            # Let's do O(ln^3) for simplicity, it's fast for small ln.
            for i in range(ln):
                for j in range(i+1, ln):
                    u, v = nodes[i], nodes[j]
                    if adj[u, v]:
                        # Common neighbors in nodes
                        common_mask = masks[u] & masks[v] & m
                        # Count bits in common_mask
                        count += bin(common_mask).count('1')
            return count // 3

        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    common_mask = masks[u] & masks[v]
                    common_mask &= ~(1 << u)
                    common_mask &= ~(1 << v)
                    k5_count += count_k3_in_mask(common_mask)
                    
        ind5_count = 0
        # Independent 5-set is K5 in complement.
        # Complement mask: ~masks[i] & ((1<<n)-1) & ~(1<<i)
        comp_masks = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp_masks[i] = (all_mask ^ masks[i]) & ~(1 << i)
            
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u, v]:
                    # u and v are non-adjacent, so they are adjacent in complement
                    common_mask = comp_masks[u] & comp_masks[v]
                    common_mask &= ~(1 << u)
                    common_mask &= ~(1 << v)
                    # Count K3 in complement subgraph induced by common_mask
                    # We need to count triangles in the complement graph.
                    # Triangle in complement (a,b,c) means adj[a,b]==0, adj[a,c]==0, adj[b,c]==0.
                    # This is equivalent to K3 in the graph defined by comp_adj.
                    # Let's reuse logic but with comp_masks and checking comp adjacency.
                    
                    # Actually, simpler: just count independent triangles in original graph.
                    # Independent triangle (a,b,c) in G means no edges between them.
                    # a,b,c in common_mask (so non-adjacent to u,v).
                    # Check if a,b,c form an independent set in G.
                    
                    nodes = []
                    mm = common_mask
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length() - 1
                        nodes.append(idx)
                        mm ^= lsb
                        
                    ln = len(nodes)
                    if ln < 3: continue
                    
                    for i in range(ln):
                        for j in range(i+1, ln):
                            u2, v2 = nodes[i], nodes[j]
                            if not adj[u2, v2]:
                                # Check common non-neighbors in nodes
                                # common_non_neighbors = nodes that are non-adjacent to u2 and v2
                                # and are in nodes
                                # mask of nodes: m
                                # non-neighbors of u2 in m: m & ~masks[u2]
                                # non-neighbors of v2 in m: m & ~masks[v2]
                                # intersection: m & ~masks[u2] & ~masks[v2]
                                # exclude u2, v2
                                cn_mask = common_mask & (~masks[u2]) & (~masks[v2])
                                cn_mask &= ~(1 << u2)
                                cn_mask &= ~(1 << v2)
                                ind5_count += bin(cn_mask).count('1')
                    # Each independent triangle is counted 3 times (once per edge in complement, i.e., non-edge in G)
                    # Wait, the loop `if not adj[u2, v2]` iterates over non-edges.
                    # An independent set {a,b,c} has 3 non-edges.
                    # For non-edge (a,b), we find c.
                    # For non-edge (a,c), we find b.
                    # For non-edge (b,c), we find a.
                    # So yes, divide by 3.
                    
        return k5_count + ind5_count // 3

    # Fallback: Paley graph on 37 vertices
    paley_37_adj = get_paley(37)
    best_n, best_edges = 37, adj_to_edges(37, paley_37_adj)

    # Strategy: 2-fold cover of Paley(19)
    q = 19
    base_adj = get_paley(q)
    n = 2 * q
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Copy 0: 0..18
    # Copy 1: 19..37
    
    # Internal edges
    for i in range(q):
        for j in range(i+1, q):
            if base_adj[i, j]:
                # Edge in Copy 0
                adj[i, j] = 1
                adj[j, i] = 1
                # Edge in Copy 1
                adj[i+q, j+q] = 1
                adj[j+q, i+q] = 1
                
    # Cross edges: Connect u in Copy 0 to v in Copy 1 if base_adj[u, v] == 1
    # Let's try this variant first.
    for i in range(q):
        for j in range(q):
            if base_adj[i, j]:
                adj[i, j+q] = 1
                adj[j+q, i] = 1

    # Verify
    bad_count = count_bad_sets(n, adj)
    if bad_count == 0:
        return n, adj_to_edges(n, adj)
        
    # If not valid, try the other variant: connect if NOT adjacent
    # Reset adj
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(q):
        for j in range(i+1, q):
            if base_adj[i, j]:
                adj[i, j] = 1
                adj[j, i] = 1
                adj[i+q, j+q] = 1
                adj[j+q, i+q] = 1
                
    for i in range(q):
        for j in range(q):
            if not base_adj[i, j] and i != j:
                adj[i, j+q] = 1
                adj[j+q, i] = 1

    bad_count = count_bad_sets(n, adj)
    if bad_count == 0:
        return n, adj_to_edges(n, adj)
        
    # If neither works, we can try to repair the first one with local search
    # But for now, let's just return the seed if the construction fails.
    # We could also try Paley 37 as the best valid graph found so far.
    
    # Let's do a quick local search on the 38-vertex graph to see if we can fix it
    # Reuse the search logic but optimized?
    # The previous search was too slow.
    # Let's just return the seed for now if the deterministic construction fails.
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
