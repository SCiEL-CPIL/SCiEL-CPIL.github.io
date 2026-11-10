import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    n = 20
    total_possible = n * (n - 1) // 2  # 190
    
    def count_common_neighbors(adj, u, v):
        """Count common neighbors of u and v using bitmasks."""
        common = adj[u] & adj[v]
        return bin(common).count('1')
    
    def is_k23_free(adj):
        """Check if graph is K2,3-free: no two vertices have 3+ common neighbors."""
        for u in range(n):
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) >= 3:
                    return False
        return True
    
    def greedy_construction():
        """Greedy: add edges in order of minimum common neighbor impact."""
        adj = [0] * n
        # Start with random edge ordering
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            # Check if adding (u,v) would create K2,3
            # For each w, check if common(u,v,w) would be >= 3 after adding
            # Actually: adding (u,v) means u and v become adjacent.
            # We need to check: for any pair (x,y), do they have >= 3 common neighbors?
            # Adding (u,v) affects common neighbor counts for pairs involving u or v.
            # 
            # Simpler check: after adding (u,v), for each w:
            #   common(u,w) increases by 1 if v is neighbor of w
            #   common(v,w) increases by 1 if u is neighbor of w
            #   common(x,y) for x,y not in {u,v} unchanged
            #
            # So we need: for all w, count_common_neighbors(adj, u, w) + (1 if v in adj[w] else 0) < 3
            #            and count_common_neighbors(adj, v, w) + (1 if u in adj[w] else 0) < 3
            
            can_add = True
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check pair (u, w)
                cuw = count_common_neighbors(adj, u, w)
                if cuw + (1 if (adj[w] >> v) & 1 else 0) >= 3:
                    can_add = False
                    break
                # Check pair (v, w)
                cvw = count_common_neighbors(adj, v, w)
                if cvw + (1 if (adj[w] >> u) & 1 else 0) >= 3:
                    can_add = False
                    break
            
            if can_add:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj
    
    def local_search(adj):
        """Try to add more edges via local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each missing edge
            for u in range(n):
                if time.time() > deadline:
                    return adj
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    # Check if we can add (u,v)
                    can_add = True
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        cuw = count_common_neighbors(adj, u, w)
                        if cuw + (1 if (adj[w] >> v) & 1 else 0) >= 3:
                            can_add = False
                            break
                        cvw = count_common_neighbors(adj, v, w)
                        if cvw + (1 if (adj[w] >> u) & 1 else 0) >= 3:
                            can_add = False
                            break
                    
                    if can_add:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        
        return adj
    
    def remove_edge_to_fix(adj, u, v, w):
        """Remove an edge to fix a K2,3 violation involving u,v with common neighbor w."""
        # We need to remove one of the edges creating the violation
        # The violation is: u and v have w as common neighbor, plus 2 others
        # Simplest: remove edge (u, w) or (v, w)
        # Try both and see which allows more edges later
        # For now, just remove (u,w)
        adj[u] &= ~(1 << w)
        adj[w] &= ~(1 << u)
    
    def fix_violations(adj):
        """Remove edges to fix all K2,3 violations."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    cnt = bin(common).count('1')
                    if cnt >= 3:
                        # Need to remove edges to reduce common neighbors
                        # Get the common neighbors
                        common_neighbors = [w for w in range(n) if (common >> w) & 1]
                        # Remove edge (u, common_neighbors[0])
                        w = common_neighbors[0]
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        changed = True
                        break
                if changed:
                    break
        return adj
    
    def count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    # Run multiple random restarts
    best_adj = None
    best_count = 0
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 12345 + 67890)
        adj = greedy_construction()
        adj = local_search(adj)
        adj = fix_violations(adj)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [x | 0 for x in adj]
    
    # Try more restarts if time permits
    restart = num_restarts
    while time.time() < deadline - 5.0:
        random.seed(restart * 98765 + 43210)
        adj = greedy_construction()
        adj = local_search(adj)
        adj = fix_violations(adj)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [x | 0 for x in adj]
        restart += 1
    
    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback: return the simple construction
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
