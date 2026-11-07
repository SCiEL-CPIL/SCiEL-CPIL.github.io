import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 15
    
    def is_valid(adj):
        deg = [bin(adj[i]).count('1') for i in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                nu = adj[u] & ~(1 << v)
                nv = adj[v] & ~(1 << u)
                s_mask = nu
                while s_mask:
                    s = (s_mask & -s_mask).bit_length() - 1
                    s_mask &= s_mask - 1
                    t_mask = nv & ~(1 << s)
                    while t_mask:
                        t = (t_mask & -t_mask).bit_length() - 1
                        t_mask &= t_mask - 1
                        if deg[s] == deg[t]:
                            return False
        return True

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask &= mask - 1
        return edges

    # Initialize with K_{7,8}
    adj = [0] * n
    for a in range(7):
        for b in range(7, 15):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    best_adj = adj[:]
    best_edges = count_edges(adj)
    
    random.seed(42)
    
    # Local Search
    # Try to add edges greedily
    def greedy_fill(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if is_valid(adj):
                            changed = True
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
        return adj

    adj = greedy_fill(adj)
    if count_edges(adj) > best_edges:
        best_edges = count_edges(adj)
        best_adj = adj[:]
        
    # Simulated Annealing / Random Restart Local Search
    # We will try to swap edges to improve edge count.
    # Since we want max edges, we prefer adding.
    # But adding might be blocked.
    # So we do: Remove 1 edge, Add 1 edge (swap) OR Remove 1 edge, Add 2 edges (if possible).
    # Simpler: Just try random swaps of (remove e1, add e2) and check if valid and edge count increases?
    # No, swap keeps edge count same.
    # We need to INCREASE edge count.
    # So we need to find a state where we can add an edge.
    # If we are stuck, we must remove an edge to create "room" (change degrees) to allow an addition.
    
    cur_adj = best_adj[:]
    cur_edges = best_edges
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
            
        # Try to add a random missing edge
        missing_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (cur_adj[u] >> v) & 1:
                    missing_edges.append((u, v))
        
        if not missing_edges:
            break # Complete graph (unlikely to be valid)
            
        u, v = random.choice(missing_edges)
        
        # Try adding (u, v)
        cur_adj[u] |= (1 << v)
        cur_adj[v] |= (1 << u)
        
        if is_valid(cur_adj):
            cur_edges += 1
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_adj = cur_adj[:]
        else:
            # Revert
            cur_adj[u] &= ~(1 << v)
            cur_adj[v] &= ~(1 << u)
            
            # If adding failed, try a swap: Remove a random existing edge, then try to add (u,v) or another edge
            # Or just perturb: Remove 1 random edge, then greedy fill
            if random.random() < 0.1: # 10% chance to perturb
                existing_edges = adj_to_edges(cur_adj)
                if existing_edges:
                    e1 = random.choice(existing_edges)
                    u1, v1 = e1
                    cur_adj[u1] &= ~(1 << v1)
                    cur_adj[v1] &= ~(1 << u1)
                    cur_edges -= 1
                    
                    # Greedy fill
                    cur_adj = greedy_fill(cur_adj)
                    cur_edges = count_edges(cur_adj)
                    
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_adj = cur_adj[:]
                        
    return adj_to_edges(best_adj)
