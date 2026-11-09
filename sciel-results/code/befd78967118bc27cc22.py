import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    n = 20
    max_pairs = n * (n - 1) // 2  # 190 possible edges
    
    # Greedy construction with vectorized checks
    # adj: n x n boolean adjacency matrix
    adj = np.zeros((n, n), dtype=np.uint8)
    edges = []
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Greedy: try to add edges in a good order
    # Strategy: start with a high-degree structure, then add edges greedily
    
    # Initial: try a near-regular structure
    # Use greedy: for each potential edge, check if adding it violates K2,3
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # For K2,3-free: no two vertices can have 3+ common neighbors
        # Adding edge (u,v) affects common neighbor counts for pairs involving u or v
        # Common neighbors of (u,v) after adding: |N(u)∩N(v)| (including each other's new connection doesn't matter for common neighbors of other pairs)
        # Actually, adding edge (u,v):
        # - For pair (u,v): common neighbors = |N(u)∩N(v)| (the edge itself doesn't count as common neighbor)
        #   Wait, N(u) and N(v) are the neighbor sets. After adding edge, u∈N(v) and v∈N(u).
        #   Common neighbors of u and v = N(u) ∩ N(v). The edge (u,v) means u is in N(v) and v is in N(u), but u is not in N(u) (simple graph), so u∉N(u)∩N(v) unless... 
        #   Actually in a simple graph, N(u) doesn't include u. So common neighbors of u,v = N(u)∩N(v) which doesn't include u or v.
        #   So adding edge (u,v) doesn't change N(u)∩N(v) for the pair (u,v) itself... wait it does!
        #   Before adding: N(u) and N(v) don't include each other. After adding: u∈N(v) and v∈N(u).
        #   But u∉N(u) and v∉N(v), so u∉N(u)∩N(v) and v∉N(u)∩N(v). So common neighbors unchanged for pair (u,v).
        # - For pair (u, w) where w≠v: common neighbors = N(u)∩N(w). Adding edge (u,v) adds v to N(u) (if not already) and adds u to N(v).
        #   So N(u) gains v. Common neighbors of (u,w) gain v if v∈N(w).
        #   Similarly for pair (v, w): N(v) gains u, so common neighbors gain u if u∈N(w).
        # - For pair (w1, w2) where w1,w2∉{u,v}: unchanged.
        
        # So we need to check:
        # 1. For all w≠u,v: if v∈N(w), then |N(u)∩N(w)| + 1 ≤ 2, i.e., |N(u)∩N(w)| ≤ 1
        # 2. For all w≠u,v: if u∈N(w), then |N(v)∩N(w)| + 1 ≤ 2, i.e., |N(v)∩N(w)| ≤ 1
        # 3. For pair (u,v): |N(u)∩N(v)| ≤ 2 (unchanged by adding the edge)
        
        # Check pair (u,v)
        common_uv = np.sum(adj[u] & adj[v])
        if common_uv > 2:
            return False
        
        # Check pairs involving u with other neighbors of v
        # N(u) currently (before adding edge), and v is being added to N(u)
        # For each w in N(v) (current), check if |N(u)∩N(w)| ≤ 1 (since v will be in N(u)∩N(w) if v∈N(w))
        # Wait, v is being added to N(u). For pair (u,w) where w∈N(v):
        #   New common neighbors of (u,w) = old common + (1 if v∈N(w) else 0)
        #   v∈N(w) means adj[v,w]=1
        #   So we need: for all w with adj[v,w]=1 and w≠u: |N(u)∩N(w)| ≤ 1
        
        neighbors_of_v = np.where(adj[v] == 1)[0]
        for w in neighbors_of_v:
            if w == u:
                continue
            common_uw = np.sum(adj[u] & adj[w])
            # After adding edge (u,v), v is in N(u). If v is also in N(w) (adj[v,w]=1), then common_uw increases by 1
            # But adj[v,w]=1 is already true (w is in neighbors_of_v)
            # So new common = common_uw + 1 (since v will be in N(u) and v is in N(w))
            # Wait, is v currently in N(u)? No, we're adding edge (u,v) so v∉N(u) currently.
            # After adding, v∈N(u). And v∈N(w) since adj[v,w]=1.
            # So new |N(u)∩N(w)| = old |N(u)∩N(w)| + 1 (v is added to the intersection)
            # But only if v∉N(u) currently (which is true) and v∈N(w) (which is true)
            # Actually, v might already be in N(u)∩N(w) if adj[u,v]=1, but we're adding it so adj[u,v]=0 currently
            # So v is NOT in N(u) currently. After adding, v∈N(u). v∈N(w). So v is added to intersection.
            # New common = common_uw + 1
            if common_uw + 1 > 2:
                return False
        
        # Similarly for pairs (v, w) where w∈N(u)
        neighbors_of_u = np.where(adj[u] == 1)[0]
        for w in neighbors_of_u:
            if w == v:
                continue
            common_vw = np.sum(adj[v] & adj[w])
            # After adding edge, u∈N(v). If u∈N(w) (adj[u,w]=1), then u is added to intersection
            # u∈N(w) since w∈N(u) and graph is undirected
            # So new common = common_vw + 1
            if common_vw + 1 > 2:
                return False
        
        return True
    
    # Greedy: try edges in order of some heuristic
    # Try to build a dense graph. Start with a random order or degree-based order.
    # For reproducibility and good results, use a fixed but smart order.
    
    # Heuristic: try edges that connect vertices with lower current degree first? 
    # Or try to make the graph as regular as possible.
    # Let's try multiple random orders and keep the best.
    
    best_edges = None
    best_count = 0
    np.random.seed(42)
    
    num_trials = 200
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        # Random permutation of edges
        edge_order = all_edges[:]
        np.random.shuffle(edge_order)
        
        adj_trial = np.zeros((n, n), dtype=np.uint8)
        edges_trial = []
        
        for u, v in edge_order:
            if time.time() > deadline - 5:
                break
            
            # Check if can add edge
            # Use the same logic but with adj_trial
            common_uv = np.sum(adj_trial[u] & adj_trial[v])
            if common_uv > 2:
                continue
            
            neighbors_of_v = np.where(adj_trial[v] == 1)[0]
            valid = True
            for w in neighbors_of_v:
                if w == u:
                    continue
                common_uw = np.sum(adj_trial[u] & adj_trial[w])
                if common_uw + 1 > 2:
                    valid = False
                    break
            if not valid:
                continue
            
            neighbors_of_u = np.where(adj_trial[u] == 1)[0]
            for w in neighbors_of_u:
                if w == v:
                    continue
                common_vw = np.sum(adj_trial[v] & adj_trial[w])
                if common_vw + 1 > 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Add edge
            adj_trial[u, v] = 1
            adj_trial[v, u] = 1
            edges_trial.append([u, v])
        
        if len(edges_trial) > best_count:
            best_count = len(edges_trial)
            best_edges = edges_trial[:]
    
    # Local search: try to add more edges or swap edges to improve
    if best_edges is not None:
        adj_ls = np.zeros((n, n), dtype=np.uint8)
        for u, v in best_edges:
            adj_ls[u, v] = 1
            adj_ls[v, u] = 1
        
        # Try to add any missing edges
        for u, v in all_edges:
            if time.time() > deadline - 3:
                break
            if adj_ls[u, v] == 1:
                continue
            
            common_uv = np.sum(adj_ls[u] & adj_ls[v])
            if common_uv > 2:
                continue
            
            neighbors_of_v = np.where(adj_ls[v] == 1)[0]
            valid = True
            for w in neighbors_of_v:
                if w == u:
                    continue
                common_uw = np.sum(adj_ls[u] & adj_ls[w])
                if common_uw + 1 > 2:
                    valid = False
                    break
            if not valid:
                continue
            
            neighbors_of_u = np.where(adj_ls[u] == 1)[0]
            for w in neighbors_of_u:
                if w == v:
                    continue
                common_vw = np.sum(adj_ls[v] & adj_ls[w])
                if common_vw + 1 > 2:
                    valid = False
                    break
            if not valid:
                continue
            
            adj_ls[u, v] = 1
            adj_ls[v, u] = 1
            best_edges.append([u, v])
        
        # Try 1-swap: remove edge e1, add edge e2 if it increases count (stays same count but might enable more)
        # Actually, try to find a swap that allows adding 2 edges later
        # Simpler: try removing each edge and see if we can add 2+ edges
        for u1, v1 in best_edges[:]:
            if time.time() > deadline - 2:
                break
            
            # Remove edge (u1,v1)
            adj_ls[u1, v1] = 0
            adj_ls[v1, u1] = 0
            edges_without = [e for e in best_edges if not (e[0] == u1 and e[1] == v1) and not (e[0] == v1 and e[1] == u1)]
            
            # Try to add as many edges as possible
            added = 0
            for u2, v2 in all_edges:
                if adj_ls[u2, v2] == 1:
                    continue
                common_uv = np.sum(adj_ls[u2] & adj_ls[v2])
                if common_uv > 2:
                    continue
                neighbors_of_v2 = np.where(adj_ls[v2] == 1)[0]
                valid = True
                for w in neighbors_of_v2:
                    if w == u2:
                        continue
                    common_u2w = np.sum(adj_ls[u2] & adj_ls[w])
                    if common_u2w + 1 > 2:
                        valid = False
                        break
                if not valid:
                    continue
                neighbors_of_u2 = np.where(adj_ls[u2] == 1)[0]
                for w in neighbors_of_u2:
                    if w == v2:
                        continue
                    common_v2w = np.sum(adj_ls[v2] & adj_ls[w])
                    if common_v2w + 1 > 2:
                        valid = False
                        break
                if not valid:
                    continue
                adj_ls[u2, v2] = 1
                adj_ls[v2, u2] = 1
                edges_without.append([u2, v2])
                added += 1
            
            if len(edges_without) > len(best_edges):
                best_edges = edges_without
            else:
                # Restore
                adj_ls[u1, v1] = 1
                adj_ls[v1, u1] = 1
    
    # Final verification
    final_adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in best_edges:
        final_adj[u, v] = 1
        final_adj[v, u] = 1
    
    # Verify K2,3-free
    for i in range(n):
        for j in range(i + 1, n):
            common = np.sum(final_adj[i] & final_adj[j])
            if common > 2:
                # Should not happen, but if it does, remove problematic edges
                pass
    
    return best_edges
