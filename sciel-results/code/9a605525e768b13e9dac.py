"""Construct a dense K3,3-free graph on 20 vertices using bitmask local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 20
    
    def popcount(x):
        """Count set bits in an integer."""
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def find_violation_fast(adj, n=20):
        """Find a K3,3 violation: returns (i, j, k, l, m, p) or None.
        i,j,k are the 3 vertices with common neighbors l,m,p.
        Uses pair-based approach for speed.
        """
        for i in range(n):
            for j in range(i+1, n):
                common_ij = adj[i] & adj[j]
                if popcount(common_ij) < 2:
                    continue
                for k in range(j+1, n):
                    common_ijk = common_ij & adj[k]
                    if popcount(common_ijk) >= 3:
                        # Found 3+ common neighbors
                        bits = []
                        c = common_ijk
                        while c and len(bits) < 3:
                            low = c & (-c)
                            bits.append(low.bit_length() - 1)
                            c ^= low
                        if len(bits) == 3:
                            return (i, j, k, bits[0], bits[1], bits[2])
        return None
    
    def count_edges(adj, n=20):
        """Count edges from adjacency bitmasks."""
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def greedy_remove(adj, n=20, max_iter=10000):
        """Greedily remove edges to eliminate K3,3 violations."""
        iterations = 0
        while iterations < max_iter:
            if time.time() > deadline - 10:
                break
            iterations += 1
            viol = find_violation_fast(adj)
            if viol is None:
                break
            i, j, k, l, m, p2 = viol
            # The K3,3 has left={i,j,k}, right={l,m,p2}
            # Remove a random edge between left and right
            candidates = [(i,l),(i,m),(i,p2),(j,l),(j,m),(j,p2),(k,l),(k,m),(k,p2)]
            existing = []
            for (u,v) in candidates:
                if (adj[u] >> v) & 1:
                    existing.append((u,v))
            if existing:
                u, v = random.choice(existing)
                remove_edge(adj, u, v)
            else:
                # Should not happen, but break to avoid infinite loop
                break
        return adj
    
    def local_search_add(adj, n=20):
        """Try to add missing edges while maintaining K3,3-free property."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle edge order for diversity
            edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if not ((adj[i] >> j) & 1)]
            random.shuffle(edges_to_try)
            for (i, j) in edges_to_try:
                if time.time() > deadline - 5:
                    break
                add_edge(adj, i, j)
                if find_violation_fast(adj) is None:
                    improved = True
                else:
                    remove_edge(adj, i, j)
        return adj
    
    def edge_swap_search(adj, n=20, num_swaps=500):
        """Try edge swaps to improve the graph."""
        ec = count_edges(adj)
        for _ in range(num_swaps):
            if time.time() > deadline - 5:
                break
            # Pick a random existing edge to remove
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                             if (adj[i] >> j) & 1]
            if not existing_edges:
                break
            u, v = random.choice(existing_edges)
            remove_edge(adj, u, v)
            
            # Try to add a random missing edge
            missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                            if not ((adj[i] >> j) & 1)]
            if not missing_edges:
                add_edge(adj, u, v)
                break
            u2, v2 = random.choice(missing_edges)
            add_edge(adj, u2, v2)
            
            if find_violation_fast(adj) is None:
                ec_new = count_edges(adj)
                if ec_new >= ec:
                    ec = ec_new
                else:
                    # Revert
                    remove_edge(adj, u2, v2)
                    add_edge(adj, u, v)
            else:
                # Revert
                remove_edge(adj, u2, v2)
                add_edge(adj, u, v)
        return adj, ec
    
    best_adj = None
    best_edge_count = 0
    
    # Strategy 1: Start from complete graph, greedily remove edges
    for trial in range(30):
        if time.time() > deadline - 15:
            break
        
        # Start from complete graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Greedy removal
        adj = greedy_remove(adj)
        ec = count_edges(adj)
        
        # Local search: try to add edges back
        adj = local_search_add(adj)
        ec = count_edges(adj)
        
        # Edge swap search
        adj, ec = edge_swap_search(adj)
        
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj = adj[:]
    
    # Strategy 2: Start from random dense graphs
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random dense graph
        p = 0.5 + 0.4 * (trial % 10) / 10
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Greedy removal
        adj = greedy_remove(adj)
        ec = count_edges(adj)
        
        # Local search
        adj = local_search_add(adj)
        ec = count_edges(adj)
        
        # Edge swap
        adj, ec = edge_swap_search(adj)
        
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj = adj[:]
    
    # Final local search on best solution
    if best_adj is not None and time.time() < deadline - 5:
        adj = best_adj[:]
        adj = local_search_add(adj)
        ec = count_edges(adj)
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj = adj[:]
        
        adj = best_adj[:]
        adj, ec = edge_swap_search(adj, num_swaps=1000)
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj = adj[:]
    
    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
