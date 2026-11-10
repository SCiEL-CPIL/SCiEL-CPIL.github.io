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
        return common.bit_count()

    def is_k23_free(adj):
        """Check if graph is K2,3-free: no two vertices have 3+ common neighbors."""
        for u in range(n):
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) >= 3:
                    return False
        return True

    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return False
        
        # If u and v already have >= 3 common neighbors, the graph is already invalid
        # for this pair, so we shouldn't be in this state. But if it happens, adding 
        # doesn't fix it, and we can't add if it's already bad.
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False

        # Adding (u,v) affects common neighbor counts for pairs involving u or v.
        # For any w != u, v:
        #   common(u, w) increases by 1 if v in N(w)
        #   common(v, w) increases by 1 if u in N(w)
        # We need new_count <= 2, so old_count + increment <= 2 => old_count <= 1 if increment is 1.
        
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check pair (u, w)
            # Increment is 1 if v is a neighbor of w
            if (adj[w] >> v) & 1:
                if (adj[u] & adj[w]).bit_count() > 1:
                    return False
            
            # Check pair (v, w)
            # Increment is 1 if u is a neighbor of w
            if (adj[w] >> u) & 1:
                if (adj[v] & adj[w]).bit_count() > 1:
                    return False
        
        return True

    def greedy_construction(adj=None):
        """Greedy: add edges in order of minimum common neighbor impact."""
        if adj is None:
            adj = [0] * n
            
        # Start with random edge ordering
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if can_add(adj, u, v):
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
                    if can_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        
        return adj

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
                    cnt = common.bit_count()
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
        return sum(adj[u].bit_count() for u in range(n)) // 2

    # Run multiple random restarts
    best_adj = None
    best_count = 0

    # Strategy 1: Random Greedy
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 12345 + 67890)
        adj = greedy_construction()
        adj = local_search(adj)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Strategy 2: Start from seed (5 K4s) and improve
    # The seed is 5 disjoint K4s.
    # Each K4 has 6 edges. Total 30 edges.
    # Let's try to improve from this structure.
    
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

    seed_adj = init_seed()
    seed_adj = local_search(seed_adj)
    cnt = count_edges(seed_adj)
    if cnt > best_count:
        best_count = cnt
        best_adj = seed_adj[:]

    # Strategy 3: More sophisticated random restarts with different shuffling
    restart = num_restarts
    while time.time() < deadline - 5.0:
        random.seed(restart * 98765 + 43210)
        adj = greedy_construction()
        adj = local_search(adj)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        restart += 1

    # Final Polish: Try to add edges to best_adj
    if best_adj is not None:
        best_adj = local_search(best_adj)
        best_count = count_edges(best_adj)
        
        # Try random removals and re-additions?
        # Simple 1-opt: remove an edge, try to add others
        for _ in range(10): # Limited time
            if time.time() > deadline:
                break
            trial_adj = best_adj[:]
            # Remove a random edge
            edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (trial_adj[u] >> v) & 1:
                        edges_list.append((u, v))
            if not edges_list:
                continue
            u, v = random.choice(edges_list)
            trial_adj[u] &= ~(1 << v)
            trial_adj[v] &= ~(1 << u)
            
            # Try to add edges
            trial_adj = local_search(trial_adj)
            cnt = count_edges(trial_adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = trial_adj[:]

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
