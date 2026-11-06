import time
import random
import copy

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def is_k23_free(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        cu = (adj[u] & adj[v]).bit_count()
        if cu > 2:
            return False
        
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[v] >> w) & 1:
                common_uw = (adj[u] & adj[w]).bit_count()
                if common_uw > 1:
                    return False
            if (adj[u] >> w) & 1:
                common_vw = (adj[v] & adj[w]).bit_count()
                if common_vw > 1:
                    return False
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    def greedy_build(seed=None):
        """Greedy random construction with optional seed for reproducibility."""
        rng = random.Random(seed)
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if is_k23_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, max_iters=10000, rng=None, time_budget=None):
        """Local search with set-based edge tracking for efficiency."""
        if rng is None:
            rng = random.Random()
        adj = list(adj)
        
        if time_budget is None:
            time_budget = 5.0
        search_start = time.time()
        
        # Build edge sets
        present = set()
        missing = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    present.add((u, v))
                else:
                    missing.add((u, v))
        
        improved = True
        while improved:
            if time.time() - search_start > time_budget:
                break
            improved = False
            
            # Try adding any missing edge
            for (u, v) in list(missing):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    present.add((u, v))
                    missing.discard((u, v))
                    improved = True
                    break
            if improved:
                continue
            
            # Try 1-swap
            found_swap = False
            if not present or not missing:
                break
            
            present_list = list(present)
            missing_list = list(missing)
            
            for _ in range(200):
                if time.time() - search_start > time_budget:
                    break
                a, b = rng.choice(present_list)
                c, d = rng.choice(missing_list)
                
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                if is_k23_free(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    present.discard((a, b))
                    present.add((c, d))
                    missing.add((a, b))
                    missing.discard((c, d))
                    improved = True
                    found_swap = True
                    break
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            
            if found_swap:
                continue
            
            # Try 2-swap
            found_2swap = False
            if len(present) < 2 or len(missing) < 2:
                break
            
            present_list = list(present)
            missing_list = list(missing)
            
            for _ in range(100):
                if time.time() - search_start > time_budget:
                    break
                (a, b), (c, d) = rng.sample(present_list, 2)
                (e, f), (g, h) = rng.sample(missing_list, 2)
                
                adj[a] &= ~(1 << b); adj[b] &= ~(1 << a)
                adj[c] &= ~(1 << d); adj[d] &= ~(1 << c)
                
                if is_k23_free(adj, e, f):
                    adj[e] |= (1 << f); adj[f] |= (1 << e)
                    if is_k23_free(adj, g, h):
                        adj[g] |= (1 << h); adj[h] |= (1 << g)
                        present.discard((a, b))
                        present.discard((c, d))
                        present.add((e, f))
                        present.add((g, h))
                        missing.add((a, b))
                        missing.add((c, d))
                        missing.discard((e, f))
                        missing.discard((g, h))
                        improved = True
                        found_2swap = True
                        break
                    else:
                        adj[e] &= ~(1 << f); adj[f] &= ~(1 << e)
                
                adj[a] |= (1 << b); adj[b] |= (1 << a)
                adj[c] |= (1 << d); adj[d] |= (1 << c)
            
            if not found_2swap:
                break
        
        return adj

    def structured_initial():
        """Create a structured initial graph using circulant + greedy fill."""
        adj = [0] * n
        
        # Start with a 5-circulant: each vertex connects to ±1, ±2, +5 mod 20
        for i in range(n):
            for d in [1, 2, 5]:
                j = (i + d) % n
                if j != i and not ((adj[i] >> j) & 1):
                    if is_k23_free(adj, i, j):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        # Fill remaining edges greedily
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        for (u, v) in all_edges:
            if not ((adj[u] >> v) & 1):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj

    def random_structured_initial(seed=None):
        """Create a random structured initial graph with random 4-regular base."""
        rng = random.Random(seed)
        adj = [0] * n
        
        # Random base: connect each vertex to ~4 random others
        for i in range(n):
            candidates = [j for j in range(n) if j != i and not ((adj[i] >> j) & 1)]
            rng.shuffle(candidates)
            for j in candidates[:4]:
                if is_k23_free(adj, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Fill remaining edges greedily in random order
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if not ((adj[u] >> v) & 1):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj

    best_adj = None
    best_count = 0

    # Try multiple initial constructions
    inits = [structured_initial()]
    for s in range(3):
        inits.append(random_structured_initial(s))
    for s in range(3):
        inits.append(greedy_build(s))
    
    for adj in inits:
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = list(adj)
            best_count = cnt
    
    # Local search on best initial
    if best_adj and time.time() < deadline - 15.0:
        rng = random.Random(42)
        time_budget = min(8.0, max(2.0, (deadline - 15.0 - time.time()) * 0.4))
        adj_ls = local_search(best_adj, max_iters=10000, rng=rng, time_budget=time_budget)
        cnt_ls = edge_count(adj_ls)
        if cnt_ls > best_count:
            best_adj = list(adj_ls)
            best_count = cnt_ls

    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 8.0:
        seed = restart_count
        adj = greedy_build(seed)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = list(adj)
            best_count = cnt
        
        if time.time() < deadline - 8.0:
            rng = random.Random(seed + 1000)
            time_budget = max(1.5, min(4.0, (deadline - 8.0 - time.time()) * 0.3))
            adj_ls = local_search(adj, max_iters=10000, rng=rng, time_budget=time_budget)
            cnt_ls = edge_count(adj_ls)
            if cnt_ls > best_count:
                best_adj = list(adj_ls)
                best_count = cnt_ls
        
        if time.time() < deadline - 8.0 and restart_count % 2 == 1:
            adj_rs = random_structured_initial(seed)
            cnt_rs = edge_count(adj_rs)
            if cnt_rs > best_count:
                best_adj = list(adj_rs)
                best_count = cnt_rs
            
            if time.time() < deadline - 8.0:
                rng = random.Random(seed + 2000)
                time_budget = max(1.5, min(4.0, (deadline - 8.0 - time.time()) * 0.3))
                adj_ls = local_search(adj_rs, max_iters=10000, rng=rng, time_budget=time_budget)
                cnt_ls = edge_count(adj_ls)
                if cnt_ls > best_count:
                    best_adj = list(adj_ls)
                    best_count = cnt_ls
        
        restart_count += 1

    if best_adj is None:
        # Fallback: simple construction
        best_adj = greedy_build()

    return adj_to_edges(best_adj)
