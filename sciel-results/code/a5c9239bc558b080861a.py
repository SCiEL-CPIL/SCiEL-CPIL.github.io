import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    range_n = range(n)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0
    
    def count_common_neighbors(adj, u, v):
        """Count common neighbors of u and v using bitmasks."""
        common = adj[u] & adj[v]
        return common.bit_count()
        
    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        We assume (u, v) is NOT currently in adj.
        Condition: For all w != u, v, 
        |N(u) \cap N(w)| + (1 if v in N(w) else 0) < 3
        |N(v) \cap N(w)| + (1 if u in N(w) else 0) < 3
        """
        # Precompute bit masks for u and v neighbors? No, just iterate w.
        # adj[u] and adj[v] do not contain each other yet.
        
        # If we add (u, v), N(u) becomes N(u) U {v}, N(v) becomes N(v) U {u}.
        
        for w in range_n:
            if w == u or w == v:
                continue
            
            # Check constraint for pair (u, w)
            # Common neighbors of u and w after adding v to u's neighborhood
            # Current common: adj[u] & adj[w]
            # New common: (adj[u] | (1<<v)) & adj[w] = (adj[u] & adj[w]) | ((1<<v) & adj[w])
            # Since v != w and v != u, and graph is simple, v is not in adj[u] currently.
            # Is v in adj[w]? Check bit.
            common_uw = adj[u] & adj[w]
            count_uw = common_uw.bit_count()
            if (adj[w] >> v) & 1:
                count_uw += 1
            if count_uw >= 3:
                return False
                
            # Check constraint for pair (v, w)
            common_vw = adj[v] & adj[w]
            count_vw = common_vw.bit_count()
            if (adj[w] >> u) & 1:
                count_vw += 1
            if count_vw >= 3:
                return False
                
        return True

    def greedy_build():
        """Build a K3,3-free graph using greedy edge addition."""
        adj = [0] * n
        edges = []
        order = all_edges[:]
        random.shuffle(order)
        
        for u, v in order:
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        return edges, adj

    def local_search(edges, adj):
        """Improve a K3,3-free graph via local search (add edges)."""
        current_edges = edges[:]
        current_adj = adj[:]
        current_count = len(current_edges)
        edge_set = set((min(e), max(e)) for e in current_edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding edges
            for u, v in all_edges:
                if (u, v) in edge_set:
                    continue
                
                if is_valid_addition(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append([u, v])
                    edge_set.add((u, v))
                    current_count += 1
                    improved = True
        
        return current_edges, current_adj, current_count

    # Phase 1: Multiple greedy builds with local search
    # Since the check is now O(N) per edge, and we have N^2 edges,
    # one pass is O(N^3) ~ 8000 ops. Very fast.
    # We can do many restarts.
    
    restarts = 0
    while time.time() < deadline - 10:
        edges, adj = greedy_build()
        edges, adj, count = local_search(edges, adj)
        
        if count > best_count:
            best_count = count
            best_edges = edges
            
        restarts += 1
        
        # Optional: Try a swap heuristic if we have time?
        # For now, rely on multiple random restarts of greedy + fill.
        
    return best_edges
