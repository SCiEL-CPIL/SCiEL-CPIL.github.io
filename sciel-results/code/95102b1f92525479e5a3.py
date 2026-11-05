import time
import random

def construct():
    n = 20
    start_time = time.time()
    # Safety margin: The evaluator kills at 120s (or 150s in some contexts, but task says 120s). 
    # We aim to finish well before. Let's use 115s as hard stop for internal logic to be safe.
    deadline = start_time + 115.0
    
    # Precompute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    # Bitmask helper for _can_add
    def can_add(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph K2,3-free.
        adj: list of bitmasks representing adjacency.
        u, v: vertices to connect. Assumes u != v and edge not present.
        """
        au = adj[u]
        av = adj[v]
        
        # 1. Check common neighbors of u and v.
        # If they already have >= 3 common neighbors, adding (u,v) creates a K2,3 
        # with any 3 of those common neighbors.
        # Actually, K2,3 requires 2 vertices (u,v) to have 3 common neighbors.
        # So if common(u,v) >= 3, it's invalid.
        common_uv = au & av
        if common_uv.bit_count() > 2:
            return False
        
        # 2. Check neighbors of v (excluding u).
        # If w is a neighbor of v, then after adding (u,v), u and w share neighbors:
        # The old common neighbors of u and w, plus v (since v is now neighbor of u and w).
        # So we need: common(u, w) + 1 <= 2  => common(u, w) <= 1.
        # Note: v is in adj[w] iff w is in adj[v] (symmetry).
        # We iterate through neighbors of v.
        w = av
        while w:
            low = w & (-w)
            w ^= low
            wi = low.bit_length() - 1
            if wi == u:
                continue
            # common neighbors of u and wi
            c = (au & adj[wi]).bit_count()
            # If wi is neighbor of v, then v is common neighbor of u and wi after adding (u,v)?
            # No. v is neighbor of wi (since wi in av). And v becomes neighbor of u.
            # So v is a common neighbor of u and wi.
            # Is v already counted in (au & adj[wi])?
            # au is adj[u] BEFORE adding v. So v is NOT in au.
            # So v is NOT in (au & adj[wi]).
            # Thus, the new common count is c + 1.
            if c + 1 > 2:
                return False
                
        # 3. Check neighbors of u (excluding v).
        # Symmetric to above.
        w = au
        while w:
            low = w & (-w)
            w ^= low
            wi = low.bit_length() - 1
            if wi == v:
                continue
            # common neighbors of v and wi
            c = (av & adj[wi]).bit_count()
            # u is neighbor of wi (since wi in au). u becomes neighbor of v.
            # So u is a common neighbor of v and wi.
            # u is not in av (before adding), so not in (av & adj[wi]).
            # New common count is c + 1.
            if c + 1 > 2:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_fill(adj):
        """
        Greedily add any possible edge until no more can be added.
        Returns the number of edges added.
        """
        added = 0
        # We loop until no progress is made.
        # To speed up, we can shuffle edges or just iterate. 
        # Given n=20, 190 edges, a simple loop is fine.
        made_progress = True
        while made_progress:
            made_progress = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if can_add(adj, u, v):
                        add_edge(adj, u, v)
                        added += 1
                        made_progress = True
        return added

    def edge_count(adj):
        count = 0
        for u in range(n):
            count += adj[u].bit_count()
        return count // 2

    def local_search(adj, deadline):
        """
        Local search: Remove a random edge, greedily refill, keep if improved.
        Also try removing 2 edges.
        """
        current_adj = adj[:]
        current_count = edge_count(current_adj)
        best_adj = current_adj[:]
        best_count = current_count
        
        rng = random.Random(42)
        
        # Precompute list of current edges for faster random selection
        def get_edges_list(adj):
            edges = []
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
            return edges

        iters = 0
        while time.time() < deadline - 2.0:
            iters += 1
            if iters > 5000: # Hard cap on iterations per local search to prevent stuck
                break
                
            current_edges = get_edges_list(current_adj)
            if not current_edges:
                break
            
            # Choose move: 1-edge removal or 2-edge removal
            if rng.random() < 0.7:
                # 1-edge removal
                if not current_edges: continue
                u, v = rng.choice(current_edges)
                old_adj = current_adj[:]
                remove_edge(current_adj, u, v)
                # Refill
                greedy_fill(current_adj)
                new_count = edge_count(current_adj)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = current_adj[:]
                else:
                    # Revert
                    current_adj = old_adj
            else:
                # 2-edge removal
                if len(current_edges) < 2: continue
                i = rng.randrange(len(current_edges))
                j = rng.randrange(len(current_edges))
                if i == j: continue
                e1, e2 = current_edges[i], current_edges[j]
                old_adj = current_adj[:]
                remove_edge(current_adj, *e1)
                remove_edge(current_adj, *e2)
                greedy_fill(current_adj)
                new_count = edge_count(current_adj)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = current_adj[:]
                else:
                    current_adj = old_adj
                    
        return best_adj, best_count

    # Main Loop
    best_global_adj = None
    best_global_count = 0
    
    # Phase 1: Randomized Greedy Initialization
    # We run many restarts.
    # Strategy: 
    # 1. Pure random order greedy.
    # 2. Random subset start, then greedy.
    
    n_restarts = 500
    # We will allocate most time to Phase 1, keeping top solutions for Phase 2.
    # But Phase 2 is expensive. Let's do 100 restarts, keep top 10, then search.
    
    # Let's do a mix: 200 fast restarts, keep top 20.
    top_solutions = [] # list of (count, adj)
    
    time_per_restart = (deadline - 30.0) / 200.0 # Reserve 30s for local search
    
    for r in range(200):
        if time.time() > deadline - 35.0:
            break
            
        adj = [0] * n
        
        # Randomized greedy: shuffle edge order
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        for u, v in edges_copy:
            if not ((adj[u] >> v) & 1):
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
        
        # Greedy fill to ensure maximal for this order
        greedy_fill(adj)
        
        count = edge_count(adj)
        
        # Maintain top 20
        top_solutions.append((count, adj))
        top_solutions.sort(key=lambda x: x[0], reverse=True)
        if len(top_solutions) > 20:
            top_solutions.pop()
        
        if count > best_global_count:
            best_global_count = count
            best_global_adj = adj[:]

    # Phase 2: Local Search on top solutions
    # Sort top solutions by count descending
    top_solutions.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate
    seen = set()
    unique_solutions = []
    for count, adj in top_solutions:
        # Create a hashable representation
        key = tuple(tuple(sorted([adj[u] for u in range(n)]))) # Not a good key
        # Better key: frozenset of edges
        edges = frozenset((u, v) for u, v in all_edges if (adj[u] >> v) & 1)
        if edges not in seen:
            seen.add(edges)
            unique_solutions.append((count, adj))
            
    # Run local search on unique top solutions
    # Allocate remaining time
    remaining_time = deadline - time.time()
    n_searches = min(len(unique_solutions), 10)
    time_per_search = remaining_time / n_searches if n_searches > 0 else 0
    
    for i in range(n_searches):
        if time.time() > deadline - 5.0:
            break
        count, adj = unique_solutions[i]
        search_deadline = time.time() + time_per_search - 1.0
        if search_deadline < time.time() + 1.0:
            search_deadline = time.time() + 1.0
            
        new_adj, new_count = local_search(adj, search_deadline)
        
        if new_count > best_global_count:
            best_global_count = new_count
            best_global_adj = new_adj[:]

    # Final Polish: Greedy fill on the best solution found
    if best_global_adj is not None:
        greedy_fill(best_global_adj)
        best_global_count = edge_count(best_global_adj)

    # Convert to list of edges [u, v]
    result = []
    if best_global_adj is not None:
        for u, v in all_edges:
            if (best_global_adj[u] >> v) & 1:
                result.append([u, v])
    else:
        # Fallback: empty graph or random
        pass

    return result
