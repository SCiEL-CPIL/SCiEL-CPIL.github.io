import time
import random
import math
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

# EVOLVE-BLOCK-START

def construct():
    """
    Maximize edges in a K2,3-free graph on 20 vertices.
    
    Strategy:
    1. Greedy with random restarts using bitmask adjacency
    2. Local search with 1-for-1 edge swaps
    3. Exhaustive 1-for-2 swap polish (remove 1 edge, add 2 new edges)
    4. Final greedy pass
    """
    n = 20
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    deadline = start_time + time_limit
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def count_edges(adj):
        return sum(adj[u].bit_count() for u in range(n)) // 2
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if adj[u] & (1 << v):
            return True
        
        # Common neighbors of u and v must be <= 2
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # For each w in N(u)\{v}, common neighbors of w and v must be <= 1
        # (since u becomes a new common neighbor of w and v)
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u] & (1 << w):
                if (adj[w] & adj[v]).bit_count() > 1:
                    return False
        
        # For each w in N(v)\{u}, common neighbors of w and u must be <= 1
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v] & (1 << w):
                if (adj[w] & adj[u]).bit_count() > 1:
                    return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    def greedy_fill(adj, order=None):
        """Greedily add edges in the given order."""
        if order is None:
            order = all_pairs
        
        for u, v in order:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
    
    def local_search_1for1(adj, max_iters=20000):
        """Local search with 1-for-1 edge swaps."""
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj[:]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try adding a random non-edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj[u] & (1 << v)):
                continue
            
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Try 1-for-1 swap: remove random incident edge, try to add (u,v)
                incident = []
                for w in range(n):
                    if w != u and (adj[u] & (1 << w)):
                        incident.append((u, w))
                    if w != v and (adj[v] & (1 << w)):
                        if w != u:
                            incident.append((v, w))
                
                if incident:
                    ru, rv = random.choice(incident)
                    remove_edge(adj, ru, rv)
                    current_edges -= 1
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                    else:
                        add_edge(adj, ru, rv)
                        current_edges += 1
        
        return best_adj, best_edges
    
    def exhaustive_1for2_swap(adj):
        """
        Exhaustive 1-for-2 swap: for each edge e to remove, try all pairs of 
        non-edges to add. If removing e allows adding both new edges, net gain +1.
        This escapes local optima where no single edge can be added.
        """
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj[:]
        
        # Get current edge list
        current_edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    current_edges_list.append((u, v))
        
        # Get non-edge list
        non_edges = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
        
        for rem_u, rem_v in current_edges_list:
            if time.time() > deadline:
                break
            
            # Remove edge (rem_u, rem_v)
            remove_edge(adj, rem_u, rem_v)
            
            # Try all pairs of non-edges
            for i in range(len(non_edges)):
                if time.time() > deadline:
                    break
                ne1_u, ne1_v = non_edges[i]
                
                # Skip if this non-edge is the removed edge
                if (ne1_u, ne1_v) == (rem_u, rem_v) or (ne1_u, ne1_v) == (rem_v, rem_u):
                    continue
                
                if not can_add_edge(adj, ne1_u, ne1_v):
                    continue
                
                # Add first non-edge
                add_edge(adj, ne1_u, ne1_v)
                
                for j in range(i + 1, len(non_edges)):
                    if time.time() > deadline:
                        break
                    ne2_u, ne2_v = non_edges[j]
                    
                    # Skip if this is the removed edge or same as first
                    if (ne2_u, ne2_v) == (rem_u, rem_v) or (ne2_u, ne2_v) == (rem_v, rem_u):
                        continue
                    if (ne2_u, ne2_v) == (ne1_u, ne1_v) or (ne2_u, ne2_v) == (ne1_v, ne1_u):
                        continue
                    
                    if can_add_edge(adj, ne2_u, ne2_v):
                        add_edge(adj, ne2_u, ne2_v)
                        # We have a valid 1-for-2 swap: net +1 edge
                        new_edges = count_edges(adj)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = adj[:]
                        # Revert the second add
                        remove_edge(adj, ne2_u, ne2_v)
                
                # Revert the first add
                remove_edge(adj, ne1_u, ne1_v)
            
            # Restore removed edge
            add_edge(adj, rem_u, rem_v)
        
        return best_adj, best_edges
    
    # --- Strategy 1: Greedy with random restarts ---
    best_adj = None
    best_edges = 0
    
    num_restarts = 300
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # Leave time for polish
            break
        
        adj = [0] * n
        random_order = all_pairs[:]
        random.shuffle(random_order)
        
        greedy_fill(adj, order=random_order)
        curr_edges = count_edges(adj)
        
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = adj[:]
    
    # --- Strategy 2: Local search on best found ---
    if best_adj is not None and time.time() < deadline - 20:
        improved_adj, improved_edges = local_search_1for1(best_adj[:], max_iters=30000)
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj = improved_adj
    
    # --- Strategy 3: More restarts with local search ---
    for restart in range(50):
        if time.time() > deadline - 25:
            break
        
        adj = [0] * n
        random_order = all_pairs[:]
        random.shuffle(random_order)
        
        greedy_fill(adj, order=random_order)
        curr_edges = count_edges(adj)
        
        if curr_edges >= best_edges - 1:  # Only polish if close to best
            improved_adj, improved_edges = local_search_1for1(adj, max_iters=5000)
            if improved_edges > best_edges:
                best_edges = improved_edges
                best_adj = improved_adj
    
    # --- Strategy 4: Exhaustive 1-for-2 swap polish ---
    if best_adj is not None and time.time() < deadline - 5:
        polished_adj, polished_edges = exhaustive_1for2_swap(best_adj[:])
        if polished_edges > best_edges:
            best_edges = polished_edges
            best_adj = polished_adj
    
    # --- Strategy 5: Final greedy pass ---
    if best_adj is not None and time.time() < deadline - 1:
        greedy_fill(best_adj, order=all_pairs)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges
    
    # --- Strategy 6: One more 1-for-2 polish if time allows ---
    if best_adj is not None and time.time() < deadline - 2:
        polished_adj, polished_edges = exhaustive_1for2_swap(best_adj[:])
        if polished_edges > best_edges:
            best_edges = polished_edges
            best_adj = polished_adj
    
    if best_adj is None:
        # Fallback: 5 disjoint K4s (20 edges)
        best_adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    add_edge(best_adj, u, v)
    
    return get_edges_from_adj(best_adj)

# EVOLVE-BLOCK-END
