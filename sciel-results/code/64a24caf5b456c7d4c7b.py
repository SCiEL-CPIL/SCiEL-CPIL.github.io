import time
import random

def construct():
    n = 16
    deadline = time.time() + 108  # safety margin before 120s limit
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    def is_triangle_free_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        return (adj[u] & adj[v]) == 0
    
    def is_3_colorable(adj):
        """Backtracking 3-coloring check with bitmasks."""
        color = [-1] * n
        
        # Order vertices by degree (highest first) for faster pruning
        order = sorted(range(n), key=lambda i: bin(adj[i]).count('1'), reverse=True)
        
        def backtrack(pos):
            if pos == n:
                return True
            i = order[pos]
            
            # Determine used colors from already-colored neighbors
            mask = adj[i]
            used = 0
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if color[j] != -1:
                    used |= (1 << color[j])
                m ^= lsb
            
            for c in range(3):
                if not (used & (1 << c)):
                    color[i] = c
                    if backtrack(pos + 1):
                        return True
            color[i] = -1
            return False
        
        return backtrack(0)
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            mask_higher = mask & ~((1 << (u + 1)) - 1)
            m = mask_higher
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                edges.append([u, v])
                m ^= lsb
        return edges
    
    def make_grotzsch_adj():
        adj = [0] * n
        # C5: 0-1-2-3-4-0
        cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Copies 5-9 for 0-4 (Mycielski construction)
        for u, v in cycle:
            adj[u] |= (1 << (5 + v))
            adj[5 + v] |= (1 << u)
            adj[v] |= (1 << (5 + u))
            adj[5 + u] |= (1 << v)
            
        # w=10 connected to 5-9
        for i in range(5):
            adj[10] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
            
        return adj
    
    def make_random_tf_adj():
        """Generate a random triangle-free graph."""
        adj = [0] * n
        random.shuffle(all_edges)
        for u, v in all_edges:
            if is_triangle_free_add(adj, u, v):
                if random.random() < 0.7:  # Accept with some probability
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    def local_search(adj, deadline):
        """Perform local search with edge additions and swaps."""
        current_adj = adj[:]
        current_edges = sum(bin(a).count('1') for a in current_adj) // 2
        
        # Get list of current edges
        current_edge_set = set()
        for u in range(n):
            mask = current_adj[u]
            mask_higher = mask & ~((1 << (u + 1)) - 1)
            m = mask_higher
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                current_edge_set.add((u, v))
                m ^= lsb
        
        # Get list of missing edges
        missing_edges = [e for e in all_edges if e not in current_edge_set]
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # Try adding edges
            random.shuffle(missing_edges)
            for u, v in missing_edges[:30]:  # Limit work per iteration
                if is_triangle_free_add(current_adj, u, v):
                    # Tentatively add
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
                    if not is_3_colorable(current_adj):
                        new_count = current_edges + 1
                        if new_count > current_edges:
                            current_edges = new_count
                            current_edge_set.add((u, v))
                            missing_edges.remove((u, v))
                            improved = True
                        else:
                            current_adj[u] &= ~(1 << v)
                            current_adj[v] &= ~(1 << u)
                    else:
                        # Revert
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                
                if time.time() >= deadline:
                    break
            
            # Try edge swaps: remove one edge, add another
            if not improved and current_edge_set:
                swap_candidates = list(current_edge_set)
                random.shuffle(swap_candidates)
                for rem_u, rem_v in swap_candidates[:20]:
                    # Remove edge
                    current_adj[rem_u] &= ~(1 << rem_v)
                    current_adj[rem_v] &= ~(1 << rem_u)
                    
                    # Try to add a different edge
                    random.shuffle(missing_edges)
                    for add_u, add_v in missing_edges[:15]:
                        if is_triangle_free_add(current_adj, add_u, add_v):
                            current_adj[add_u] |= (1 << add_v)
                            current_adj[add_v] |= (1 << add_u)
                            
                            if not is_3_colorable(current_adj):
                                # Swap is valid (still 4-chromatic, same edge count or we can try to improve)
                                # Actually swap keeps same count, so only useful if it enables future improvements
                                # For now, accept if it doesn't break things
                                current_edge_set.remove((rem_u, rem_v))
                                current_edge_set.add((add_u, add_v))
                                missing_edges.remove((add_u, add_v))
                                missing_edges.append((rem_u, rem_v))
                                improved = True
                                break
                            else:
                                # Revert
                                current_adj[add_u] &= ~(1 << add_v)
                                current_adj[add_v] &= ~(1 << add_u)
                    
                    # Revert removal if no successful swap
                    if (rem_u, rem_v) in current_edge_set:
                        pass  # Was added back
                    else:
                        current_adj[rem_u] |= (1 << rem_v)
                        current_adj[rem_v] |= (1 << rem_u)
                    
                    if time.time() >= deadline:
                        break
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return current_adj, current_edges
    
    # Main search loop with multiple strategies
    strategies = []
    
    # Strategy 1: Grötzsch seed
    grotzsch_adj = make_grotzsch_adj()
    strategies.append(("grotzsch", grotzsch_adj))
    
    # Strategy 2-5: Random triangle-free seeds
    for i in range(4):
        rand_adj = make_random_tf_adj()
        if not is_3_colorable(rand_adj):
            strategies.append((f"random_{i}", rand_adj))
    
    # Run local search on each strategy
    for name, adj in strategies:
        if time.time() >= deadline:
            break
        
        # Deep copy
        adj_copy = adj[:]
        result_adj, result_edges = local_search(adj_copy, deadline)
        
        if result_edges > best_count:
            best_count = result_edges
            best_edges = get_edges_from_adj(result_adj)
    
    # Final greedy expansion on best result
    if best_edges and time.time() < deadline:
        final_adj = [0] * n
        for u, v in best_edges:
            final_adj[u] |= (1 << v)
            final_adj[v] |= (1 << u)
        
        # One more round of greedy edge addition
        missing = [e for e in all_edges if e not in set(map(tuple, best_edges))]
        random.shuffle(missing)
        for u, v in missing:
            if is_triangle_free_add(final_adj, u, v):
                final_adj[u] |= (1 << v)
                final_adj[v] |= (1 << u)
        
        new_edges = get_edges_from_adj(final_adj)
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges
    
    return best_edges
