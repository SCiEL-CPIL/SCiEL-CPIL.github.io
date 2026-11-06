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
        common_with_v = [0] * n
        for c in range(n):
            if c != v and c != u:
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

    def polish(adj_bits, edge_set, deadline):
        """
        Repeatedly try to add any missing edge in multiple passes until no more can be added.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in all_pairs:
                if time.time() > deadline:
                    return
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if (adj_bits[u] >> v) & 1:
                    continue
                if not has_c6(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edge_set.add((u, v))
                    edge_set.add((v, u))
                    improved = True
        # Return list of edges
        edges = []
        for u, v in edge_set:
            if u < v:
                edges.append([u, v])
        return edges

    def one_swap_search(adj_bits, edge_set, deadline):
        """
        Try removing one edge at a time, then try to add other edges.
        Keep the best result.
        """
        current_edges = []
        for u, v in edge_set:
            if u < v:
                current_edges.append([u, v])
        
        best_local_count = len(current_edges)
        best_local_adj = adj_bits[:]
        best_local_edge_set = edge_set.copy()
        
        edge_list = [(u, v) for u, v in edge_set if u < v]
        
        for rem_u, rem_v in edge_list:
            if time.time() > deadline:
                break
            
            # Remove edge
            adj_bits[rem_u] &= ~(1 << rem_v)
            adj_bits[rem_v] &= ~(1 << rem_u)
            edge_set.discard((rem_u, rem_v))
            edge_set.discard((rem_v, rem_u))
            
            # Try to add other edges
            added_any = False
            for u, v in all_pairs:
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
                    added_any = True
            
            # Count edges
            current_count = len([e for e in edge_set if e[0] < e[1]])
            
            if current_count > best_local_count:
                best_local_count = current_count
                best_local_adj = adj_bits[:]
                best_local_edge_set = edge_set.copy()
            
            # Restore original state
            adj_bits[rem_u] |= (1 << rem_v)
            adj_bits[rem_v] |= (1 << rem_u)
            edge_set.add((rem_u, rem_v))
            edge_set.add((rem_v, rem_u))
            
            # If we found an improvement, restart the search from the improved state
            if best_local_count > len(current_edges):
                # Update current state to best found
                adj_bits = best_local_adj[:]
                edge_set = best_local_edge_set.copy()
                current_edges = []
                for u, v in edge_set:
                    if u < v:
                        current_edges.append([u, v])
                # Continue searching from improved state
                break
        
        return best_local_adj, best_local_edge_set, best_local_count

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
            # Degree-based (high degree first)
            if current_edge_set:
                degrees = [bin(x).count('1') for x in current_adj]
                current_order = sorted(all_pairs, key=lambda p: degrees[p[0]] + degrees[p[1]], reverse=True)
            else:
                current_order = all_pairs[:]
                random.shuffle(current_order)
        elif strategy == 2:
            current_order = all_pairs[:]
            random.shuffle(current_order)
        else:
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
            
    # Final polish and 1-swap search on the best solution
    if time.time() < deadline:
        # Polish first
        polished_edges = polish(best_adj, best_edge_set, deadline)
        best_count = len(polished_edges)
        best_edges = polished_edges[:]
        
        # Then try 1-swap
        if time.time() < deadline:
            new_adj, new_edge_set, new_count = one_swap_search(best_adj, best_edge_set, deadline)
            if new_count > best_count:
                best_count = new_count
                best_adj = new_adj
                best_edge_set = new_edge_set
                best_edges = []
                for u, v in best_edge_set:
                    if u < v:
                        best_edges.append([u, v])
                # Polish again
                if time.time() < deadline:
                    polished_edges = polish(best_adj, best_edge_set, deadline)
                    best_count = len(polished_edges)
                    best_edges = polished_edges[:]
    
    return best_edges

# EVOLVE-BLOCK-END
