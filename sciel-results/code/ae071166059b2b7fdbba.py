import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 20
    max_edges = n * (n - 1) // 2
    
    # Precompute all pairs
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def can_add_k23_free(adj, u, v):
        # adj is list of int masks
        # Check if adding (u, v) creates a K2,3
        # 1. Check pair (u, v): common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 2. Check pairs (u, x) for x in N(v)
        # For each x in N(v), v becomes a new common neighbor of u and x.
        # So we need current common neighbors of u and x to be <= 1.
        mask_v = adj[v]
        x = 0
        m = mask_v
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            m ^= lsb
            if x == u:
                continue
            if (adj[u] & adj[x]).bit_count() > 1:
                return False
        
        # 3. Check pairs (v, y) for y in N(u)
        # For each y in N(u), u becomes a new common neighbor of v and y.
        # So we need current common neighbors of v and y to be <= 1.
        y = 0
        m = adj[u]
        while m:
            lsb = m & -m
            y = lsb.bit_length() - 1
            m ^= lsb
            if y == v:
                continue
            if (adj[v] & adj[y]).bit_count() > 1:
                return False
        
        return True

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            for v in range(u + 1, n):
                if (mask >> v) & 1:
                    edges.append([u, v])
        return edges

    best_edge_count = 0
    best_adj = [0] * n

    def greedy_init():
        adj = [0] * n
        shuffled_pairs = pairs[:]
        random.shuffle(shuffled_pairs)
        
        for u, v in shuffled_pairs:
            if can_add_k23_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        count = sum(a.bit_count() for a in adj) // 2
        return adj, count

    def random_neighbor(adj, u):
        mask = adj[u]
        if not mask:
            return -1
        neighbors = [v for v in range(n) if (mask >> v) & 1]
        return random.choice(neighbors)

    while time.time() < deadline:
        adj, current_count = greedy_init()
        
        # Local Search
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        no_improve = 0
        
        for step in range(50000):
            if time.time() > deadline:
                break
            
            if current_count > best_edge_count:
                best_edge_count = current_count
                best_adj = adj[:]
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 2000:
                break
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Add edge
                u, v = random.choice(pairs)
                if not (adj[u] & (1 << v)):
                    if can_add_k23_free(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_count += 1
            elif move_type < 0.7:
                # Remove edge
                u = random.randint(0, n - 1)
                v = random_neighbor(adj, u)
                if v != -1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_count -= 1
            else:
                # Swap edge: remove (u, v), add (x, y)
                u = random.randint(0, n - 1)
                v = random_neighbor(adj, u)
                if v != -1:
                    # Remove (u, v)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_count -= 1
                    # Try to add a random non-edge
                    x, y = random.choice(pairs)
                    if not (adj[x] & (1 << y)) and x != y:
                        if can_add_k23_free(adj, x, y):
                            adj[x] |= (1 << y)
                            adj[y] |= (1 << x)
                            current_count += 1
                        else:
                            pass

            T *= cooling
            if T < T_min:
                T = T_min
    
    # Polish the best solution
    adj = best_adj[:]
    improved = True
    while improved:
        improved = False
        for u, v in pairs:
            if not (adj[u] & (1 << v)) and can_add_k23_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                improved = True
    
    # Combine edges
    edges = get_edges_from_adj(adj)
    
    return edges
