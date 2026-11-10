import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    total_possible = n * (n - 1) // 2
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(adj, u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
            
        # 2. For all w in N(v), pair (u,w) gains v as common neighbor.
        #    So current common(u,w) must be <= 1.
        #    w iterates over neighbors of v
        w = adj[v]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if popcount(adj[u] & adj[wi]) > 1:
                return False
                
        # 3. For all w in N(u), pair (v,w) gains u as common neighbor.
        #    So current common(v,w) must be <= 1.
        w = adj[u]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if popcount(adj[v] & adj[wi]) > 1:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2
        
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                lsb = v_mask & (-v_mask)
                v_mask ^= lsb
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
        return edges

    def greedy_build(adj):
        # Randomly add edges while maintaining K2,3-free property
        order = list(range(n * (n - 1) // 2))
        random.shuffle(order)
        # Map index to (u,v) pair
        pair_map = {}
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                pair_map[idx] = (u, v)
                idx += 1
                
        for i in order:
            if time.time() > deadline:
                break
            u, v = pair_map[i]
            if can_add(adj, u, v):
                add_edge(adj, u, v)

    def local_search(adj, max_iters=5000):
        # Try to improve by adding edges
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try to add a random missing edge
            for _ in range(100):
                if time.time() > deadline:
                    return adj
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if (adj[u] >> v) & 1:
                    continue
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
                    break
        return adj

    # Multi-start approach
    best_adj = [0] * n
    best_edges = 0
    
    num_starts = 50
    for s in range(num_starts):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        greedy_build(adj)
        adj = local_search(adj)
        
        edges_count = count_edges(adj)
        if edges_count > best_edges:
            best_edges = edges_count
            best_adj = adj[:]
            
    # Convert best_adj to edge list
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
