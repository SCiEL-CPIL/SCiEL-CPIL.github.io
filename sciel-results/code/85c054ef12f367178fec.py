import random
import math
import time

def construct():
    n = 18
    
    # Precompute bits
    all_mask = (1 << n) - 1
    
    # State: adj_mask list of length n
    # adj_mask[i] has bit j set if edge (i,j) exists
    
    def is_k4_free(adj_mask):
        # Check for each edge (u,v), if common[u][v] has an edge.
        for u in range(n):
            mask_u = adj_mask[u]
            # Iterate neighbors v of u
            v = 0
            while v < n:
                if mask_u & (1 << v):
                    # Edge (u,v) exists
                    common = mask_u & adj_mask[v]
                    # Check if common contains an edge
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & -c
                        w_idx = lsb.bit_length() - 1
                        if adj_mask[w_idx] & common:
                            has_edge = True
                            break
                        c ^= lsb
                    if has_edge:
                        return False
                v += 1
        return True

    def max_triangle_free_size(adj_mask):
        # Branch and Bound with better heuristic
        best_size = 0
        
        # Precompute common neighbors for all pairs
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u][v] = adj_mask[u] & adj_mask[v]
        
        # Precompute degrees in the "potential" graph? 
        # We need to pick a vertex from p_mask.
        # Heuristic: pick v in p_mask with max degree in the subgraph induced by p_mask?
        # Or pick v that maximizes the number of vertices it conflicts with?
        
        def dfs(c_mask, p_mask):
            nonlocal best_size
            
            # Pruning
            if bin(c_mask).count('1') + bin(p_mask).count('1') <= best_size:
                return
            
            if p_mask == 0:
                if bin(c_mask).count('1') > best_size:
                    best_size = bin(c_mask).count('1')
                return
            
            # Heuristic: Pick vertex v in p_mask
            # Let's pick the one with the highest "conflict potential"
            # i.e., max number of neighbors in p_mask?
            # Simpler: just pick the first one? 
            # Let's try picking the one with max degree in p_mask.
            
            # To find max degree in p_mask:
            # deg[v] = popcount(adj_mask[v] & p_mask)
            # This is O(n) per step.
            
            best_v = -1
            max_deg = -1
            temp_p = p_mask
            while temp_p:
                lsb = temp_p & -temp_p
                v_idx = lsb.bit_length() - 1
                deg = bin(adj_mask[v_idx] & p_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    best_v = v_idx
                temp_p ^= lsb
            
            if best_v == -1:
                return
                
            v_lsb = 1 << best_v
            p_mask_without_v = p_mask ^ v_lsb
            
            # Option 1: Include v
            # Compute invalid mask
            invalid = 0
            c = c_mask
            while c:
                lsb_u = c & -c
                u_idx = lsb_u.bit_length() - 1
                if adj_mask[u_idx] & v_lsb:
                    invalid |= common[best_v][u_idx]
                c ^= lsb_u
            
            new_c_mask = c_mask | v_lsb
            new_p_mask = p_mask_without_v & ~invalid
            
            # Additional pruning: if new_p_mask is empty, update best
            # and recurse
            
            dfs(new_c_mask, new_p_mask)
            
            # Option 2: Exclude v
            dfs(c_mask, p_mask_without_v)

        dfs(0, all_mask)
        return best_size

    # Simulated Annealing
    best_score = n # Max possible is n
    
    # Run multiple SA instances to find a good graph
    # Time budget: 110 seconds
    
    start_time = time.time()
    time_limit = 110.0
    
    for run_id in range(10):
        random.seed(42 + run_id)
        
        # Initialize with a random K4-free graph
        # Start with empty graph and add random edges if K4-free?
        # Or start with Turan and randomize?
        # Let's start with a random permutation and a random threshold?
        # Better: Start with Turan T(18,3) and apply random flips.
        
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        adj_mask = [0]*n
        for i in range(3):
            for j in range(i+1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
        
        # Randomize
        for _ in range(100):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            bit = (1 << v)
            # Try to flip
            adj_mask[u] ^= bit
            adj_mask[v] ^= (1 << u)
            if not is_k4_free(adj_mask):
                adj_mask[u] ^= bit
                adj_mask[v] ^= (1 << u)
        
        current_adj = adj_mask[:]
        current_score = max_triangle_free_size(current_adj)
        
        if current_score < best_score:
            best_score = current_score
            best_adj = current_adj[:]
            
        T_start = 10.0
        T_end = 0.01
        iterations = 5000
        
        for i in range(iterations):
            if time.time() - start_time > time_limit:
                break
                
            T = T_start * (T_end / T_start) ** (i / iterations)
            
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            
            bit = (1 << v)
            current_adj[u] ^= bit
            current_adj[v] ^= (1 << u)
            
            if not is_k4_free(current_adj):
                current_adj[u] ^= bit
                current_adj[v] ^= (1 << u)
                continue
                
            new_score = max_triangle_free_size(current_adj)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        current_adj[u] ^= bit
                        current_adj[v] ^= (1 << u)
                else:
                    current_score = new_score

        if best_score <= 9:
            break

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        m = best_adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
            
    return edges
