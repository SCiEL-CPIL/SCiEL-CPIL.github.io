import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    
    # Best known solution so far
    best_edges = 0
    best_adj = [0] * n
    
    # Initialize with the seed solution (5 K4s) to have a valid starting point
    def init_seed():
        adj = [0] * n
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def can_add(adj, u, v):
        # If already connected
        if (adj[u] >> v) & 1:
            return False
            
        # Check common neighbors of u and v
        if (adj[u] & adj[v]).bit_count() > 2:
            return False

        # Check effect on other pairs
        for w in range(n):
            if w == u or w == v:
                continue
            
            if (adj[u] >> w) & 1:
                if (adj[v] & adj[w]).bit_count() > 1:
                    return False
            
            if (adj[v] >> w) & 1:
                if (adj[u] & adj[w]).bit_count() > 1:
                    return False

        return True

    def saturate(adj):
        # Try to add edges until no more can be added
        changed = True
        while changed:
            changed = False
            # Get list of non-edges
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        non_edges.append((u, v))
                
            if not non_edges:
                break
                
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj

    def can_swap(adj, e1, e2):
        """Check if removing e1=(u1,v1) and adding e2=(u2,v2) maintains K2,3-free."""
        u1, v1 = e1
        u2, v2 = e2
        
        # e2 must be a non-edge
        if (adj[u2] >> v2) & 1:
            return False
        
        # Remove e1 temporarily
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        
        # Now try to add e2
        ok = can_add(adj, u2, v2)
        
        # Restore e1
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)
        
        return ok

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Initial Setup
    best_adj = init_seed()
    best_adj = saturate(best_adj)
    best_edges = count_edges(best_adj)

    # Local Search with Perturbation and Edge Swaps
    # We will iteratively:
    # 1. Take the best known graph
    # 2. Apply a random perturbation (remove 1 edge, or swap 1 edge)
    # 3. Re-saturate
    # 4. If better, update best
    
    # Precompute all possible edges for swap candidates
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        # Create candidate
        cand_adj = best_adj[:]
        
        # Get current edges
        edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if (cand_adj[u] >> v) & 1:
                    edges_list.append((u, v))
                        
        # Choose perturbation type
        if random.random() < 0.4 and edges_list:
            # Edge swap: remove one existing edge, try to add a different one
            # Pick a random existing edge to remove
            rem_u, rem_v = random.choice(edges_list)
            cand_adj[rem_u] &= ~(1 << rem_v)
            cand_adj[rem_v] &= ~(1 << rem_u)
        elif edges_list:
            # Simple edge removal
            u, v = random.choice(edges_list)
            cand_adj[u] &= ~(1 << v)
            cand_adj[v] &= ~(1 << u)
        else:
            # If no edges, start fresh
            cand_adj = [0] * n

        # Saturate the candidate
        cand_adj = saturate(cand_adj)
        
        cand_edges = count_edges(cand_adj)
        
        if cand_edges > best_edges:
            best_edges = cand_edges
            best_adj = cand_adj[:]
            
        # Occasionally, try a completely fresh random start to escape global basins
        if random.random() < 0.05:
            fresh_adj = [0] * n
            fresh_adj = saturate(fresh_adj)
            fresh_edges = count_edges(fresh_adj)
            if fresh_edges > best_edges:
                best_edges = fresh_edges
                best_adj = fresh_adj[:]

    return get_edges_from_adj(best_adj)
