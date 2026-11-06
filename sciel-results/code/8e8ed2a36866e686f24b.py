"""Optimized K2,3-free graph construction on 20 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def count_common_neighbors(adj, u, v):
        return bin(adj[u] & adj[v]).count('1')
        
    def can_add_edge(adj, u, v):
        if u == v or ((adj[u] >> v) & 1):
            return False
        
        # 1. The pair (u, v) itself must have < 3 common neighbors
        if count_common_neighbors(adj, u, v) >= 3:
            return False
            
        # 2. Adding (u, v) makes u a common neighbor of (v, w) for all w in N(u)
        #    So for all w in N(u), we need current common neighbors of (v, w) to be <= 1
        #    (so that adding u makes it at most 2)
        #    Wait, if current is 2, adding u makes it 3. Invalid.
        #    So current must be <= 1.
        neighbors_u = []
        mask_u = adj[u]
        for w in range(n):
            if (mask_u >> w) & 1:
                neighbors_u.append(w)
                
        for w in neighbors_u:
            if w == v:
                continue
            if count_common_neighbors(adj, v, w) >= 2:
                return False
                
        # 3. Similarly, adding (u, v) makes v a common neighbor of (u, w) for all w in N(v)
        neighbors_v = []
        mask_v = adj[v]
        for w in range(n):
            if (mask_v >> w) & 1:
                neighbors_v.append(w)
                
        for w in neighbors_v:
            if w == u:
                continue
            if count_common_neighbors(adj, u, w) >= 2:
                return False
                
        return True
        
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def get_edge_count(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2
        
    def is_valid_graph(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if ((adj[u] >> v) & 1):
                    if count_common_neighbors(adj, u, v) >= 3:
                        return False
        return True

    best_adj = [0] * n
    best_count = 0
    rng = random.Random(42)
    
    # Strategy: Multiple random greedy starts + Local Search
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
            
        adj = [0] * n
        edges_list = all_edges[:]
        rng.shuffle(edges_list)
        
        # Greedy Phase
        for u, v in edges_list:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
                
        # Local Search Phase: Try to add edges (should be none if greedy was max, but maybe order dependent)
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
        
        current_count = get_edge_count(adj)
        
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
            
        # Local Search Phase: Swaps (Remove 1, Add 2)
        # We try to find a beneficial swap.
        # To be efficient, we only do this if we have time.
        if time.time() < deadline - 10:
            for u, v in all_edges:
                if ((adj[u] >> v) & 1):
                    remove_edge(adj, u, v)
                    
                    added = 0
                    added_edges = []
                    # Try to add edges in random order to escape local optima?
                    # Or just fixed order.
                    for u2, v2 in all_edges:
                        if u2 == u and v2 == v:
                            continue
                        if not ((adj[u2] >> v2) & 1) and can_add_edge(adj, u2, v2):
                            add_edge(adj, u2, v2)
                            added_edges.append((u2, v2))
                            added += 1
                            if added == 2:
                                break
                            
                    if added > 1:
                        new_count = current_count - 1 + added
                        if new_count > best_count:
                            best_adj = adj[:]
                            best_count = new_count
                            current_count = new_count
                        # Note: If added > 1, we keep the state. 
                        # If added <= 1, we rollback.
                    else:
                        for u2, v2 in added_edges:
                            remove_edge(adj, u2, v2)
                        add_edge(adj, u, v)
                        
                    if time.time() > deadline - 5:
                        break

    # Final verification
    if not is_valid_graph(best_adj):
        best_adj = [0] * n
        for u, v in all_edges:
            if can_add_edge(best_adj, u, v):
                add_edge(best_adj, u, v)
                
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result_edges.append([u, v])
                
    return result_edges

# EVOLVE-BLOCK-END
