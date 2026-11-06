import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0

    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph K2,3-free.
        adj is a list of bitmasks.
        """
        # 1. u and v must not have > 2 common neighbors
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # 2. For all w in N(u) \ {v}, |N(v) \cap N(w)| <= 1
        # N(u) is adj[u]. We iterate over set bits.
        mask_u = adj[u]
        # Exclude v
        mask_u_ex_v = mask_u & ~(1 << v)
        # Iterate w in mask_u_ex_v
        m = mask_u_ex_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            # Check |N(v) & N(w)| <= 1
            if (adj[v] & adj[w]).bit_count() > 1:
                return False
            m ^= lsb
            
        # 3. For all x in N(v) \ {u}, |N(u) \cap N(x)| <= 1
        mask_v = adj[v]
        mask_v_ex_u = mask_v & ~(1 << u)
        m = mask_v_ex_u
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            if (adj[u] & adj[x]).bit_count() > 1:
                return False
            m ^= lsb
            
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

    def greedy_build(seed):
        rng = random.Random(seed)
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, rng, time_budget):
        adj = list(adj)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            # Try to add edges
            # Iterate over all missing edges
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_valid_add(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try 1-swap
                # Pick a random present edge to remove, then try to add a missing one
                present_edges = []
                for u in range(n):
                    mask = adj[u]
                    v = u + 1
                    while v < n:
                        if (mask >> v) & 1:
                            present_edges.append((u, v))
                        v += 1
                
                if not present_edges:
                    break
                    
                # Try a few random swaps
                found_swap = False
                for _ in range(100):
                    u, v = rng.choice(present_edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Try to add any missing edge
                    added = False
                    for x in range(n):
                        for y in range(x + 1, n):
                            if not ((adj[x] >> y) & 1):
                                if is_valid_add(adj, x, y):
                                    adj[x] |= (1 << y)
                                    adj[y] |= (1 << x)
                                    added = True
                                    break
                        if added:
                            break
                    
                    if added:
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Revert
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                
                if not found_swap:
                    # Try 2-swap or just stop if no improvement
                    # A simple 2-swap is complex to implement efficiently without precomputed conflicts.
                    # Given the fast check, we might just restart if stuck.
                    break

        return adj

    best_adj = None
    best_count = 0

    # Multiple restarts
    for i in range(50):
        if time.time() > deadline - 5.0:
            break
        adj = greedy_build(i)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = list(adj)
            best_count = cnt
        
        # Local search
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(i * 1000 + 42)
        budget = min(2.0, max(0.5, (deadline - 5.0 - time.time()) * 0.1))
        if budget > 0.1:
            adj_ls = local_search(adj, rng, budget)
            cnt_ls = edge_count(adj_ls)
            if cnt_ls > best_count:
                best_adj = list(adj_ls)
                best_count = cnt_ls

    if best_adj is None:
        best_adj = greedy_build(0)

    return adj_to_edges(best_adj)
