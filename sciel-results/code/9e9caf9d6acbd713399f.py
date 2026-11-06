import time
import random
import sys

def construct():
    deadline = time.time() + 110  # Safety margin
    n = 40
    best_edges = []
    best_count = 0
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_c6(adj_bits, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        Returns True if a C6 is created, False otherwise.
        """
        mask_v = adj_bits[v]
        # Precompute common neighbors of c and v for all c
        # This is O(n) bit ops
        common_with_v = [0] * n
        for c in range(n):
            if c != v and c != u: # c cannot be u or v in the path u-a-b-c-d-v
                common_with_v[c] = adj_bits[c] & mask_v
        
        a_candidates = adj_bits[u] & ~(1 << v)
        
        while a_candidates:
            a_bit = a_candidates & -a_candidates
            a = a_bit.bit_length() - 1
            a_candidates -= a_bit
            
            b_candidates = adj_bits[a] & ~(1 << u) & ~(1 << v)
            
            while b_candidates:
                b_bit = b_candidates & -b_candidates
                b = b_bit.bit_length() - 1
                b_candidates -= b_bit
                
                c_candidates = adj_bits[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                
                while c_candidates:
                    c_bit = c_candidates & -c_candidates
                    c = c_bit.bit_length() - 1
                    c_candidates -= c_bit
                    
                    forbidden = (1 << u) | (1 << a) | (1 << b) | (1 << c)
                    
                    if common_with_v[c] & ~forbidden:
                        return True
        return False

    def greedy_build(order, adj_bits, edge_set, deadline):
        """
        Greedily add edges from order to adj_bits if no C6 is formed.
        """
        added_edges = []
        for u, v in order:
            if time.time() > deadline:
                break
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            if (adj_bits[u] >> v) & 1:
                continue
            
            if not has_c6(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                added_edges.append([u, v])
        return added_edges

    def try_strategy(order, adj_bits, edge_set, deadline):
        added = greedy_build(order, adj_bits, edge_set, deadline)
        return len(added), added, adj_bits, edge_set

    random.seed(42)
    
    # Strategy 1: Start with 8 K5s
    base_adj = [0] * n
    base_edges = []
    base_edge_set = set()
    for k in range(8):
        start = k * 5
        for a in range(5):
            for b in range(a + 1, 5):
                u = start + a
                v = start + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
                base_edges.append([u, v])
                base_edge_set.add((u, v))
                base_edge_set.add((v, u))
                
    best_count = len(base_edges)
    best_edges = base_edges[:]
    best_adj = base_adj[:]
    best_edge_set = base_edge_set.copy()
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        current_adj = [0] * n
        current_edge_set = set()
        
        # Choose a starting point
        if attempts % 3 == 0:
            # Start with K5s
            current_adj = best_adj[:]
            current_edge_set = best_edge_set.copy()
        else:
            # Start empty
            pass
            
        # Choose an ordering strategy
        strategy = attempts % 4
        
        if strategy == 0:
            # Random order
            current_order = all_pairs[:]
            random.shuffle(current_order)
        elif strategy == 1:
            # Degree-based (high degree first) - but degrees are 0 initially if empty
            # So for empty start, this is random.
            # For non-empty start, we can sort by current degree
            if current_edge_set:
                degrees = [0]*n
                for u, v in current_edge_set:
                    if u < v: # avoid double count in set iteration? set has both (u,v) and (v,u)
                        pass
                # Recalculate degrees from adj_bits
                degrees = [bin(x).count('1') for x in current_adj]
                # Sort pairs by sum of degrees of endpoints
                current_order = sorted(all_pairs, key=lambda p: degrees[p[0]] + degrees[p[1]], reverse=True)
            else:
                current_order = all_pairs[:]
                random.shuffle(current_order)
        elif strategy == 2:
            # Random restart of the best solution found so far?
            # Let's just do random order but maybe bias towards edges in best_edges
            current_order = all_pairs[:]
            # Put edges from best_edges first?
            best_edge_list = set(tuple(e) for e in best_edges)
            best_edge_list_inv = set((v,u) for u,v in best_edge_list)
            # This is complex. Let's just shuffle.
            random.shuffle(current_order)
        else:
            # Reverse random?
            current_order = all_pairs[:]
            random.shuffle(current_order)
            current_order.reverse()
            
        # Run greedy
        current_edges = greedy_build(current_order, current_adj, current_edge_set, deadline)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            best_adj = current_adj[:]
            best_edge_set = current_edge_set.copy()
            
        # Local improvement: Try to add remaining edges in a different order
        # The greedy_build already iterates through all pairs in 'order'.
        # But if we start with K5s, some edges are already present.
        # The greedy_build skips present edges.
        # So it effectively tries to add missing edges in the order specified.
        
        # If we didn't start with K5s, we might have missed some edges that could be added
        # if we tried a different order for the *remaining* edges.
        # But greedy_build iterates through ALL pairs in 'order'.
        # So it tries every possible edge once.
        # If an edge is rejected, it's rejected.
        # A second pass with a different order might allow it if the graph state changed?
        # No, the graph state only grows.
        # If an edge was rejected because of a C6, adding more edges won't remove that C6.
        # So a second pass on the same graph is useless.
        
        # However, if we started with K5s, the graph state is different.
        
    return best_edges

# EVOLVE-BLOCK-END
