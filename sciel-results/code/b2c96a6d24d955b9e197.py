import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    n = 18
    best_score = 18  # Worst case
    best_edges = None
    
    # Helper to check K4-free
    def is_k4_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j]
                k_bits = common
                while k_bits:
                    lsb = k_bits & -k_bits
                    k = lsb.bit_length() - 1
                    k_bits ^= lsb
                    if adj[k] & (common & ~(1 << k)):
                        return False
        return True

    # Helper to find max triangle-free induced subgraph size using B&B
    def max_tf_size(adj):
        best = 0
        
        # Precompute: for each vertex v, and each potential S_mask,
        # we'll compute on the fly
        
        def dfs_mask(S_mask, C_mask):
            nonlocal best
            s_count = bin(S_mask).count('1')
            c_count = bin(C_mask).count('1')
            if s_count + c_count <= best:
                return
            if C_mask == 0:
                if s_count > best:
                    best = s_count
                return
            
            # Pick vertex with max degree in C (better pruning)
            v = -1
            max_deg = -1
            tmp = C_mask
            while tmp:
                lsb = tmp & -tmp
                u = lsb.bit_length() - 1
                deg = bin(adj[u] & C_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v = u
                tmp ^= lsb
            
            lsb = 1 << v
            C_mask_rest = C_mask ^ lsb
            
            # Branch 1: Include v
            S_mask_prime = S_mask | lsb
            
            new_C_mask = 0
            u_mask = C_mask_rest
            while u_mask:
                u_lsb = u_mask & -u_mask
                u = u_lsb.bit_length() - 1
                u_mask ^= u_lsb
                
                if adj[v] & u_lsb:
                    if adj[v] & adj[u] & S_mask:
                        continue
                
                new_C_mask |= u_lsb
            
            dfs_mask(S_mask_prime, new_C_mask)
            
            # Branch 2: Exclude v
            dfs_mask(S_mask, C_mask_rest)
        
        initial_C = (1 << n) - 1
        dfs_mask(0, initial_C)
        return best

    # Local search: try to improve a K4-free graph by edge modifications
    def local_search(adj, max_iter=1000):
        n_local = n
        current_score = max_tf_size(adj)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Try adding an edge
            improved = False
            edges_to_try = [(i, j) for i in range(n_local) for j in range(i+1, n_local) 
                           if not (adj[i] & (1 << j))]
            random.shuffle(edges_to_try)
            
            for i, j in edges_to_try:
                # Check if adding (i,j) keeps K4-free
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
                common = adj[i] & adj[j]
                k_bits = common
                has_k4 = False
                while k_bits:
                    lsb = k_bits & -k_bits
                    k = lsb.bit_length() - 1
                    k_bits ^= lsb
                    if adj[k] & (common & ~(1 << k)):
                        has_k4 = True
                        break
                
                if has_k4:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    continue
                
                new_score = max_tf_size(adj)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
            
            if improved:
                continue
            
            # Try removing an edge
            edges_to_try = [(i, j) for i in range(n_local) for j in range(i+1, n_local) 
                           if (adj[i] & (1 << j))]
            random.shuffle(edges_to_try)
            
            for i, j in edges_to_try:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                
                new_score = max_tf_size(adj)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            
            if not improved:
                break
        
        return current_score, adj

    # Generate initial K4-free graph with higher edge density
    def generate_k4_free_graph(edge_prob=0.4):
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for i, j in edges:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            common = adj[i] & adj[j]
            k_bits = common
            has_k4 = False
            while k_bits:
                lsb = k_bits & -k_bits
                k = lsb.bit_length() - 1
                k_bits ^= lsb
                if adj[k] & (common & ~(1 << k)):
                    has_k4 = True
                    break
            
            if has_k4:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
        
        return adj

    # Search Loop
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        
        # Generate random K4-free graph
        adj = generate_k4_free_graph(edge_prob=0.4)
        
        # Evaluate
        score = max_tf_size(adj)
        
        # Local search to improve
        score, adj = local_search(adj, max_iter=200)
        
        if score < best_score:
            best_score = score
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
            
            if best_score <= 10:
                break

    if best_edges is None:
        # Fallback to seed
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    return best_edges
