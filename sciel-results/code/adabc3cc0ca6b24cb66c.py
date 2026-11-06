import random
import math

def construct():
    n = 16
    
    # Helper to convert edge set to bitmask adjacency
    def edges_to_bitmasks(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_score_and_worst_from_bitmasks(adj):
        deg = [0] * n
        for i in range(n):
            deg[i] = adj[i].bit_count()
        
        max_sum = 0
        worst_triangles = []
        # Iterate over edges (u, v) with u < v
        for u in range(n):
            # Only consider v > u to avoid duplicates
            mask_u = adj[u] & ((1 << n) - 1) & ~((1 << (u+1)) - 1)
            while mask_u:
                lsb = mask_u & -mask_u
                v = lsb.bit_length() - 1
                mask_u ^= lsb
                
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                # Exclude u and v themselves (they are not in common anyway as no self-loops)
                # Iterate through common neighbors w
                mask_w = common
                while mask_w:
                    lsb_w = mask_w & -mask_w
                    w = lsb_w.bit_length() - 1
                    mask_w ^= lsb_w
                    
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
                        worst_triangles = [(u, v, w)]
                    elif s == max_sum:
                        worst_triangles.append((u, v, w))
                        
        return max_sum, worst_triangles

    def bitmask_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u] & ~((1 << (u+1)) - 1) # only v > u
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                mask ^= lsb
                edges.append([u, v])
        return edges

    # Initialize with K_8,8 + (0,1)
    initial_edges = set()
    for u in range(8):
        for v in range(8, 16):
            initial_edges.add((u, v))
    initial_edges.add((0, 1))
    
    adj = edges_to_bitmasks(initial_edges)
    current_score, worst_tris = get_score_and_worst_from_bitmasks(adj)
    
    # Precompute all possible edges as a list of (u, v) tuples
    all_edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Simulated Annealing
    T = 4.0
    T_min = 0.01
    cooling = 0.997
    steps = 100000 # More steps, but each is faster
    
    best_score = current_score
    best_adj = adj[:]
    
    no_improve_count = 0
    
    for step in range(steps):
        if T < T_min:
            break
            
        # Select edge to remove
        if worst_tris and random.random() < 0.8:
            tri = random.choice(worst_tris)
            u, v, w = tri
            e1 = (min(u,v), max(u,v))
            e2 = (min(u,w), max(u,w))
            e3 = (min(v,w), max(v,w))
            e_remove = random.choice([e1, e2, e3])
        else:
            # Pick a random existing edge
            # We need a list of current edges for random selection
            # Reconstructing list every step is slow. 
            # Optimization: Keep a list of edges in parallel? 
            # Or just pick random u, v and check if edge exists.
            # Given N=16, picking random u, v is fast.
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if (1 << v) & adj[u]:
                    e_remove = (min(u,v), max(u,v))
                    break
        
        # Select edge to add
        # Pick random non-edge
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if not ((1 << v) & adj[u]):
                e_add = (min(u,v), max(u,v))
                break
        
        # Apply swap
        u_rm, v_rm = e_remove
        u_ad, v_ad = e_add
        
        new_adj = adj[:]
        # Remove
        new_adj[u_rm] &= ~(1 << v_rm)
        new_adj[v_rm] &= ~(1 << u_rm)
        # Add
        new_adj[u_ad] |= (1 << v_ad)
        new_adj[v_ad] |= (1 << u_ad)
        
        new_score, new_worst_tris = get_score_and_worst_from_bitmasks(new_adj)
        delta = new_score - current_score
        
        if delta < 0 or (delta == 0 and random.random() < 0.5) or random.random() < math.exp(-delta / T):
            adj = new_adj
            current_score = new_score
            worst_tris = new_worst_tris
            
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        T *= cooling
        
        # Restart if stuck
        if no_improve_count > 5000:
            # Perturb
            for _ in range(10):
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v: continue
                    if (1 << v) & adj[u]:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        break
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v: continue
                    if not ((1 << v) & adj[u]):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        break
            current_score, worst_tris = get_score_and_worst_from_bitmasks(adj)
            no_improve_count = 0
            T = max(T, 2.0) # Reheat
            
        if best_score <= 23:
            break

    # Return best found
    final_edges = bitmask_to_edges(best_adj)
    return {'n': n, 'edges': final_edges}
