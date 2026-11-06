import random
import time

def popcount(x):
    return bin(x).count('1')

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_valid_addition(adj, u, v):
        # Check if adding edge (u, v) keeps the graph K2,3-free
        # 1. Common neighbors of u and v must be < 3
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False
        
        # 2. For any w in N(v), w is a neighbor of v. 
        #    Adding (u,v) means v is in N(u).
        #    If w is in N(v), then v is in N(w) (symmetry).
        #    So v becomes a common neighbor of u and w.
        #    Thus, common neighbors of (u, w) must currently be < 2.
        neighbors_v = adj[v]
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = (lsb.bit_length() - 1)
            if w != u:
                common_uw = adj[u] & adj[w]
                if popcount(common_uw) >= 2:
                    return False
            neighbors_v ^= lsb
            
        # 3. Similarly for w in N(u)
        neighbors_u = adj[u]
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = (lsb.bit_length() - 1)
            if w != v:
                common_vw = adj[v] & adj[w]
                if popcount(common_vw) >= 2:
                    return False
            neighbors_u ^= lsb
            
        return True

    best_edges = []
    
    for _ in range(1000): # Try multiple restarts
        if time.time() > deadline:
            break
            
        # Initialize adjacency bitmasks
        adj = [0] * n
        
        # Greedy construction
        # Random order of edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Local Search / Hill Climbing
        # Try to add any missing edge
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Shuffle to avoid bias
            missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
            random.shuffle(missing_edges)
            for u, v in missing_edges:
                if time.time() > deadline:
                    break
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        
        # Count edges
        edge_count = sum(popcount(adj[i]) for i in range(n)) // 2
        if edge_count > len(best_edges):
            # Extract edges
            current_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = (lsb.bit_length() - 1)
                    if i < j:
                        current_edges.append([i, j])
                    mask ^= lsb
            best_edges = current_edges
            
    return best_edges
