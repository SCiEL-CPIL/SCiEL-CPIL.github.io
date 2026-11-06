"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n = 40
    best_edges = []
    best_count = 0
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_c6(adj_bits, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        This is true if there is a simple path of length 5 from u to v in the current graph.
        Path: u-a-b-c-d-v
        """
        # u and v are not adjacent in current graph (we check non-edges)
        
        # Iterate a in N(u)
        a_candidates = adj_bits[u] & ~(1 << v)
        
        while a_candidates:
            a_bit = a_candidates & -a_candidates
            a = a_bit.bit_length() - 1
            a_candidates -= a_bit
            
            # Iterate b in N(a) \ {u, v}
            b_candidates = adj_bits[a] & ~(1 << u) & ~(1 << v)
            
            while b_candidates:
                b_bit = b_candidates & -b_candidates
                b = b_bit.bit_length() - 1
                b_candidates -= b_bit
                
                # Iterate c in N(b) \ {u, a, v}
                c_candidates = adj_bits[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                
                while c_candidates:
                    c_bit = c_candidates & -c_candidates
                    c = c_bit.bit_length() - 1
                    c_candidates -= c_bit
                    
                    # Check if there exists d such that:
                    # d in N(c) AND d in N(v)
                    # d not in {u, a, b, c}
                    mask_c = adj_bits[c]
                    mask_v = adj_bits[v]
                    d_candidates = mask_c & mask_v
                    forbidden = (1 << u) | (1 << a) | (1 << b) | (1 << c)
                    
                    if d_candidates & ~forbidden:
                        return True
        return False

    def greedy_build(order, adj_bits, edge_set):
        """
        Greedily add edges from order to adj_bits if no C6 is formed.
        Returns the list of added edges.
        """
        added_edges = []
        for u, v in order:
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            # Only add if not already adjacent
            if (adj_bits[u] >> v) & 1:
                continue
            
            if not has_c6(adj_bits, u, v):
                # Add edge
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                added_edges.append([u, v])
        return added_edges

    random.seed(42)
    
    # Initial baseline: 8 disjoint K5s
    # Vertices 0-4, 5-9, ..., 35-39
    base_adj = [0] * n
    base_edges = []
    for k in range(8):
        start = k * 5
        for a in range(5):
            for b in range(a + 1, 5):
                u = start + a
                v = start + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
                base_edges.append([u, v])
    
    best_count = len(base_edges)
    best_edges = base_edges[:]
    best_adj = base_adj[:]
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random restart
        current_adj = [0] * n
        current_edge_set = set()
        
        # Start with a random subset of the baseline or empty?
        # Starting empty allows finding different structures.
        # Starting with baseline allows local improvement.
        # Let's try starting empty for diversity, but bias towards dense regions.
        
        # Random order of all pairs
        current_order = all_pairs[:]
        random.shuffle(current_order)
        
        # Greedy add
        current_edges = greedy_build(current_order, current_adj, current_edge_set)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            best_adj = current_adj[:]
            
        # Local Search: Try to add more edges or swap
        # Given time constraints, simple restarts with fast check are often best.
        # But we can do a quick local pass: try to add any remaining edges.
        # The greedy pass already did this in random order.
        # A second pass with different order might catch some.
        
        # To maximize density, we can try "repair" or "swap" but it's complex.
        # Let's just do many fast greedy restarts.
        
    return best_edges
# EVOLVE-BLOCK-END
