import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # 1. Filter vertices
    valid_vertices = []
    for u in range(1, 257):
        s = 2 * u
        r = int(math.isqrt(s))
        if r * r != s:
            valid_vertices.append(u)
    
    n = len(valid_vertices)
    
    # Map original values to indices 0..n-1
    # val_to_idx = {v: i for i, v in enumerate(valid_vertices)} # Not strictly needed if we just use indices
    
    # 2. Build adjacency masks
    adj = [0] * n
    
    # Precompute squares
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)
        
    for i in range(n):
        ui = valid_vertices[i]
        mask = 0
        for j in range(n):
            if i == j: continue
            uj = valid_vertices[j]
            if ui + uj in squares:
                mask |= (1 << j)
        adj[i] = mask
        
    best_size = 0
    best_mask = 0
    
    def popcount(x):
        return bin(x).count('1')
        
    def get_indices(mask):
        idxs = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            idxs.append(idx)
            m ^= lsb
        return idxs

    # Greedy coloring bound for MIS
    # We want to bound the size of MIS in the subgraph induced by 'candidates'.
    # This is equivalent to bounding the size of Max Clique in the complement subgraph.
    # Greedy coloring of the complement gives an upper bound on the clique size.
    
    def greedy_color_bound(candidates):
        # Get vertices in candidates
        verts = get_indices(candidates)
        if not verts:
            return 0
            
        # Order by degree in complement (descending)
        # Degree in complement = (len(verts) - 1) - degree_in_G
        # degree_in_G for v in verts = popcount(adj[v] & candidates)
        
        def comp_deg(v):
            return (len(verts) - 1) - popcount(adj[v] & candidates)
            
        verts.sort(key=comp_deg, reverse=True)
        
        colors = {}
        num_colors = 0
        # For each vertex, assign smallest color not used by its neighbors in complement
        # Neighbors in complement are those NOT in adj[v]
        
        for v in verts:
            # Find colors used by complement neighbors
            used_colors = set()
            for u in verts:
                if u == v:
                    continue
                # Check if u and v are connected in G
                if adj[v] & (1 << u):
                    # Connected in G, so NOT connected in Complement
                    pass
                else:
                    # Not connected in G, so CONNECTED in Complement
                    if u in colors:
                        used_colors.add(colors[u])
                        
            # Find smallest available color
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > num_colors:
                num_colors = c
                
        return num_colors

    def search(candidates, current_size, current_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
        
        # Basic Pruning
        if current_size + popcount(candidates) <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Strong Pruning using Greedy Coloring Bound
        # The bound is an upper bound on the number of vertices we can add.
        bound = greedy_color_bound(candidates)
        if current_size + bound <= best_size:
            return
            
        # Select vertex with max degree in induced subgraph G
        cand_indices = get_indices(candidates)
        max_deg = -1
        v = -1
        for idx in cand_indices:
            deg = popcount(adj[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = idx
                
        if v == -1:
            return
            
        # Branch 1: Include v
        new_candidates_include = candidates & ~(adj[v] | (1 << v))
        search(new_candidates_include, current_size + 1, current_mask | (1 << v))
        
        # Branch 2: Exclude v
        new_candidates_exclude = candidates & ~(1 << v)
        search(new_candidates_exclude, current_size, current_mask)

    initial_candidates = (1 << n) - 1
    search(initial_candidates, 0, 0)
    
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(valid_vertices[i])
            
    if not result:
        result = [1]
        
    return result
