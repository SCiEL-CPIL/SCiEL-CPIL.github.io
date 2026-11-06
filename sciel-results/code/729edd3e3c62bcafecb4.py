import time
import random
import sys

def construct():
    n = 20
    # Safety margin: 120s limit, use 110s for computation
    deadline = time.time() + 110
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    
    # Helper to check if edge (u, v) can be added to adj
    # adj is a list of bitmasks
    def can_add(adj, u, v):
        # 1. Check common neighbors of u and v
        # If |N(u) & N(v)| > 2, invalid
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # 2. Check neighbors of u
        # For each w in N(u), adding v to N(u) means v becomes a common neighbor of u and w
        # IF v is already in N(w). Since w in N(u), (u,w) is edge.
        # We need to check if v is in N(w). If yes, common neighbors of u and w increase by 1.
        # So we need |N(u) & N(w)|_old <= 1.
        # Wait, if v is NOT in N(w), then v is not a common neighbor.
        # The common neighbors of u and w are vertices x such that (u,x) and (w,x) are edges.
        # Adding (u,v) adds v to N(u).
        # So v becomes a common neighbor of u and w IF v is in N(w).
        # So we check: for all w in N(u), if v in N(w), then |N(u) & N(w)| <= 1.
        
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            mask_u ^= lsb
            
            # If v is connected to w
            if adj[w] & (1 << v):
                # v will be a new common neighbor of u and w
                # Current common neighbors count must be <= 1
                if (adj[u] & adj[w]).bit_count() > 1:
                    return False
        
        # 3. Check neighbors of v
        # Symmetric check
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            mask_v ^= lsb
            
            # If u is connected to w
            if adj[w] & (1 << u):
                # u will be a new common neighbor of v and w
                # Current common neighbors count must be <= 1
                if (adj[v] & adj[w]).bit_count() > 1:
                    return False
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_construct():
        adj = [0] * n
        edges = []
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            if can_add(adj, u, v):
                add_edge(adj, u, v)
                edges.append((u, v))
        
        return adj, edges

    def local_search_add(adj, edges):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to avoid bias
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            
            for u, v in edge_list:
                if adj[u] & (1 << v):
                    continue
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    edges.append((u, v))
                    improved = True
        
        return adj, edges

    def local_search_swap_1_2(adj, edges):
        """Remove 1 edge, try to add 2."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            if not edges:
                break
            
            # Pick random edge to remove
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Save state
            old_adj = adj[:]
            old_edges = edges[:]
            
            remove_edge(adj, u, v)
            edges.pop(idx)
            
            added = 0
            # Try to add 2 edges
            # We can try all missing edges
            missing_edges = [e for e in all_edges if not (adj[e[0]] & (1 << e[1]))]
            random.shuffle(missing_edges)
            
            for a, b in missing_edges:
                if added >= 2:
                    break
                if can_add(adj, a, b):
                    add_edge(adj, a, b)
                    edges.append((a, b))
                    added += 1
            
            if added < 2:
                # Revert
                adj[:] = old_adj
                edges[:] = old_edges
            else:
                improved = True
        
        return adj, edges

    def local_search_swap_2_3(adj, edges):
        """Remove 2 edges, try to add 3."""
        improved = True
        # Limit iterations for 2-3 swap as it is more expensive
        max_iters = 200
        while improved and time.time() < deadline and max_iters > 0:
            improved = False
            max_iters -= 1
            if len(edges) < 2:
                break
            
            # Pick 2 random edges to remove
            i1 = random.randint(0, len(edges) - 1)
            i2 = random.randint(0, len(edges) - 2)
            if i2 >= i1:
                i2 += 1
            
            e1 = edges[i1]
            e2 = edges[i2]
            
            old_adj = adj[:]
            old_edges = edges[:]
            
            remove_edge(adj, e1[0], e1[1])
            remove_edge(adj, e2[0], e2[1])
            # Remove from list (careful with indices)
            edges.pop(i1)
            edges.pop(i2 if i2 < i1 else i2 - 1)
            
            added = 0
            missing_edges = [e for e in all_edges if not (adj[e[0]] & (1 << e[1]))]
            random.shuffle(missing_edges)
            
            for a, b in missing_edges:
                if added >= 3:
                    break
                if can_add(adj, a, b):
                    add_edge(adj, a, b)
                    edges.append((a, b))
                    added += 1
            
            if added < 3:
                # Revert
                adj[:] = old_adj
                edges[:] = old_edges
            else:
                improved = True
        
        return adj, edges

    def full_local_search(adj, edges):
        # Sequence of moves
        adj, edges = local_search_add(adj, edges)
        adj, edges = local_search_swap_1_2(adj, edges)
        adj, edges = local_search_add(adj, edges)
        adj, edges = local_search_swap_2_3(adj, edges)
        adj, edges = local_search_add(adj, edges)
        return adj, edges

    best_edges = []
    best_count = 0
    
    # Multiple restarts
    while time.time() < deadline:
        adj, edges = greedy_construct()
        adj, edges = full_local_search(adj, edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
            # If we found a very good solution, do extra polishing
            if best_count >= 120 and time.time() < deadline - 10:
                adj_best = [0] * n
                for u, v in best_edges:
                    adj_best[u] |= (1 << v)
                    adj_best[v] |= (1 << u)
                
                # Aggressive polish
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    adj_best, edges_best = local_search_add(adj_best, best_edges[:])
                    adj_best, edges_best = local_search_swap_1_2(adj_best, edges_best)
                    adj_best, edges_best = local_search_swap_2_3(adj_best, edges_best)
                    adj_best, edges_best = local_search_add(adj_best, edges_best)
                    
                    if len(edges_best) > best_count:
                        best_count = len(edges_best)
                        best_edges = edges_best[:]
                    else:
                        break # No improvement, stop polishing this instance

    return [[u, v] for u, v in best_edges]
