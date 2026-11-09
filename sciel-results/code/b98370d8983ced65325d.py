import random
import time

def construct():
    n = 20
    deadline = time.time() + 140  # Safety margin
    
    def bit_count(x):
        return bin(x).count("1")
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    def is_valid_add(adj, u, v):
        # 1. Common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if bit_count(common_uv) >= 3:
            return False
        
        # 2. u' in adj[v], check common(u, u') >= 2
        # adj[v] contains neighbors of v.
        # We need u' != u (u is not in adj[v] since edge not added yet? 
        # If u is in adj[v], edge exists. We only check missing edges.
        # So u not in adj[v].
        neighbors_v = adj[v]
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            u_prime = lsb.bit_length() - 1
            neighbors_v ^= lsb
            common_u_up = adj[u] & adj[u_prime]
            if bit_count(common_u_up) >= 2:
                return False
        
        # 3. v' in adj[u], check common(v, v') >= 2
        neighbors_u = adj[u]
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            v_prime = lsb.bit_length() - 1
            neighbors_u ^= lsb
            common_v_vp = adj[v] & adj[v_prime]
            if bit_count(common_v_vp) >= 2:
                return False
        
        # 4. Check pairs in common_uv
        # common_uv has < 3 bits set.
        if bit_count(common_uv) == 2:
            # Get the two bits
            c1 = common_uv & -common_uv
            c2 = common_uv ^ c1
            u_prime = c1.bit_length() - 1
            v_prime = c2.bit_length() - 1
            common_up_vp = adj[u_prime] & adj[v_prime]
            if bit_count(common_up_vp) >= 3:
                return False
                
        return True

    def greedy_search():
        adj = [0] * n
        random.shuffle(all_edges)
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Local search: try to add any missing edge
        # We can iterate until no improvement
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not (adj[u] >> v) & 1:
                    if is_valid_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        
        # Count edges
        count = sum(bit_count(adj[i]) for i in range(n)) // 2
        return count, adj

    # Run multiple restarts
    while time.time() < deadline:
        count, adj = greedy_search()
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            # Store edges
            best_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
            
    return [[u, v] for u, v in best_edges]
