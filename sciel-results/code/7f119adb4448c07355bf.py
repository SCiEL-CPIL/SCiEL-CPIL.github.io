import time
import random
import copy

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def is_k23_free(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # Check common(u, v) after adding:
        cu = (adj[u] & adj[v]).bit_count()
        if cu > 2:
            return False
        
        # For each w != u, v:
        for w in range(n):
            if w == u or w == v:
                continue
            w_in_v = (adj[v] >> w) & 1
            w_in_u = (adj[u] >> w) & 1
            if w_in_v:
                # v becomes common neighbor of u and w
                common_uw = (adj[u] & adj[w]).bit_count()
                if common_uw > 1:
                    return False
            if w_in_u:
                # u becomes common neighbor of v and w
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

    def local_search(adj, max_iters=5000, rng=None, time_budget=None):
        """Local search: try adding edges, then try 1-swap and 2-swap moves."""
        if rng is None:
            rng = random.Random()
        adj = list(adj)
        improved = True
        iter_count = 0
        
        if time_budget is None:
            time_budget = 5.0
        search_start = time.time()
        
        while improved and iter_count < max_iters:
            if time.time() - search_start > time_budget:
                break
            improved = False
            iter_count += 1
            
            # Try adding any missing edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_k23_free(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            
            # Try 1-swap: remove (a,b), add (c,d)
            found_swap = False
            present_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
            missing_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
            
            if not present_edges or not missing_edges:
                break
                
            for _ in range(100):
                if time.time() - search_start > time_budget:
                    break
                a, b = rng.choice(present_edges)
                c, d = rng.choice(missing_edges)
                if (a, b) == (c, d) or (a, b) == (d, c):
                    continue
                
                # Remove (a, b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                # Try adding (c, d)
                if is_k23_free(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    improved = True
                    found_swap = True
                    # Update edge lists
                    present_edges.append((c, d))
                    missing_edges.remove((c, d))
                    if (a, b) in present_edges:
                        present_edges.remove((a, b))
                    missing_edges.append((a, b))
                    break
                else:
                    # Restore (a, b)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            
            if found_swap:
                continue
            
            # Try 2-swap: remove (a,b) and (c,d), add (e,f) and (g,h)
            found_2swap = False
            if len(present_edges) < 2 or len(missing_edges) < 2:
                break
                
            for _ in range(50):
                if time.time() - search_start > time_budget:
                    break
                (a, b), (c, d) = rng.sample(present_edges, 2)
                (e, f), (g, h) = rng.sample(missing_edges, 2)
                
                # Check no overlap
                if len({(a,b),(c,d),(e,f),(g,h)}) < 4:
                    continue
                
                # Remove (a,b) and (c,d)
                adj[a] &= ~(1 << b); adj[b] &= ~(1 << a)
                adj[c] &= ~(1 << d); adj[d] &= ~(1 << c)
                
                # Try adding (e,f) and (g,h)
                if is_k23_free(adj, e, f):
                    adj[e] |= (1 << f); adj[f] |= (1 << e)
                    if is_k23_free(adj, g, h):
                        adj[g] |= (1 << h); adj[h] |= (1 << g)
                        improved = True
                        found_2swap = True
                        # Update edge lists
                        present_edges.extend([(e, f), (g, h)])
                        missing_edges.remove((e, f))
                        missing_edges.remove((g, h))
                        if (a, b) in present_edges:
                            present_edges.remove((a, b))
                        if (c, d) in present_edges:
                            present_edges.remove((c, d))
                        missing_edges.extend([(a, b), (c, d)])
                        break
                    else:
                        # Undo (e,f)
                        adj[e] &= ~(1 << f); adj[f] &= ~(1 << e)
                
                # Restore (a,b) and (c,d)
                adj[a] |= (1 << b); adj[b] |= (1 << a)
                adj[c] |= (1 << d); adj[d] |= (1 << c)
            
            if not found_2swap and not found_swap:
                break
        
        return adj

    def structured_initial():
        """Create a structured initial graph using a balanced bipartite-like construction."""
        adj = [0] * n
        
        # Use a construction based on projective plane or similar
        # Partition into two sets of 10, create bipartite graph with controlled degrees
        left = list(range(10))
        right = list(range(10, 20))
        
        # Create a bipartite graph where each left vertex connects to 4 right vertices
        # and each right vertex connects to 4 left vertices, with no two left vertices
        # sharing more than 2 right neighbors (and vice versa)
        
        # Use a 10x10 matrix with row/col sum 4 and pairwise intersection <= 2
        # This is equivalent to a 2-(10,4,2) design or similar
        
        # Simple approach: use cyclic shifts
        for i in range(10):
            for j in range(4):
                right_idx = (i * 4 + j) % 10
                u = left[i]
                v = right[right_idx]
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Add edges within left and right sets if possible
        for i in range(10):
            for j in range(i + 1, 10):
                u, v = left[i], left[j]
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        for i in range(10):
            for j in range(i + 1, 10):
                u, v = right[i], right[j]
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj

    def random_structured_initial(seed=None):
        """Create a random structured initial graph."""
        rng = random.Random(seed)
        adj = [0] * n
        
        # Random bipartite construction with controlled density
        left = list(range(10))
        right = list(range(10, 20))
        
        # Randomly assign connections with probability ~0.4
        for u in left:
            for v in right:
                if rng.random() < 0.4:
                    if is_k23_free(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Add random within-set edges
        for i in range(10):
            for j in range(i + 1, 10):
                if rng.random() < 0.3:
                    u, v = left[i], left[j]
                    if is_k23_free(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        for i in range(10):
            for j in range(i + 1, 10):
                if rng.random() < 0.3:
                    u, v = right[i], right[j]
                    if is_k23_free(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        return adj

    best_adj = None
    best_count = 0

    # Start with structured initial
    adj = structured_initial()
    cnt = edge_count(adj)
    if cnt > best_count:
        best_adj = list(adj)
        best_count = cnt
    
    # Local search on structured initial
    if time.time() < deadline - 10.0:
        adj_ls = local_search(adj, max_iters=5000, time_budget=5.0)
        cnt_ls = edge_count(adj_ls)
        if cnt_ls > best_count:
            best_adj = list(adj_ls)
            best_count = cnt_ls

    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 5.0:
        seed = restart_count
        adj = greedy_build(seed)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = list(adj)
            best_count = cnt
        
        # Local search
        if time.time() < deadline - 5.0:
            rng = random.Random(seed + 1000)
            time_budget = max(1.0, min(3.0, (deadline - 5.0 - time.time()) * 0.5))
            adj_ls = local_search(adj, max_iters=5000, rng=rng, time_budget=time_budget)
            cnt_ls = edge_count(adj_ls)
            if cnt_ls > best_count:
                best_adj = list(adj_ls)
                best_count = cnt_ls
        
        # Also try random structured initial
        if time.time() < deadline - 5.0 and restart_count % 2 == 1:
            adj_rs = random_structured_initial(seed)
            cnt_rs = edge_count(adj_rs)
            if cnt_rs > best_count:
                best_adj = list(adj_rs)
                best_count = cnt_rs
            
            if time.time() < deadline - 5.0:
                rng = random.Random(seed + 2000)
                time_budget = max(1.0, min(3.0, (deadline - 5.0 - time.time()) * 0.5))
                adj_ls = local_search(adj_rs, max_iters=5000, rng=rng, time_budget=time_budget)
                cnt_ls = edge_count(adj_ls)
                if cnt_ls > best_count:
                    best_adj = list(adj_ls)
                    best_count = cnt_ls
        
        restart_count += 1

    if best_adj is None:
        # Fallback: simple construction
        best_adj = greedy_build()

    return adj_to_edges(best_adj)
