import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    n = 20
    total_possible = n * (n - 1) // 2  # 190

    def can_add_fast(adj, u, v):
        """Fast check if adding edge (u,v) keeps the graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return False
        
        # Check u-v pair itself
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False

        # For each w != u,v:
        #   If v in N(w), then common(u,w) increases by 1, so need old common(u,w) <= 1
        #   If u in N(w), then common(v,w) increases by 1, so need old common(v,w) <= 1
        
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check pair (u, w): if v is neighbor of w, common(u,w) increases
            if (adj[w] >> v) & 1:
                if (adj[u] & adj[w]).bit_count() > 1:
                    return False
            
            # Check pair (v, w): if u is neighbor of w, common(v,w) increases
            if (adj[w] >> u) & 1:
                if (adj[v] & adj[w]).bit_count() > 1:
                    return False
        
        return True

    def greedy_construction():
        """Greedy: add edges with priority to high-degree vertex pairs."""
        adj = [0] * n
        
        # Generate all edges with random tie-breaking
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        # Sort by a heuristic: prefer pairs where both vertices have low degree
        # (more likely to be addable)
        def edge_priority(e):
            u, v = e
            return (adj[u].bit_count() + adj[v].bit_count(), random.random())
        
        # We'll do multiple passes since degrees change
        remaining = set(edges)
        for _ in range(3):  # 3 passes
            if time.time() > deadline:
                break
            # Re-sort remaining edges by current degree sum
            remaining_sorted = sorted(remaining, key=edge_priority)
            for u, v in remaining_sorted:
                if time.time() > deadline:
                    break
                if (u, v) not in remaining:
                    continue
                if can_add_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    remaining.discard((u, v))
        
        return adj

    def local_search(adj):
        """Try to add more edges via local search - try ALL addable edges per pass."""
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
                    if can_add_fast(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        
        return adj

    def count_edges(adj):
        return sum(adj[u].bit_count() for u in range(n)) // 2

    # Run multiple random restarts
    best_adj = None
    best_count = 0

    # Strategy 1: Random Greedy with multiple restarts
    num_restarts = 200
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

    # Strategy 3: More restarts with different seeds
    restart = num_restarts
    while time.time() < deadline - 10.0:
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
        
        # 1-opt: remove an edge, try to add others
        for _ in range(20):
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
