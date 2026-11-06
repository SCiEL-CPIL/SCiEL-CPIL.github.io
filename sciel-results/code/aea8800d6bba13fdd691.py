import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 255
    # 1. Generate candidates
    candidates = [v for v in range(1, n + 1) if v % 3 == 0 or v % 5 == 0 or v % 17 == 0]
    m = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = val_to_idx[255]
    
    # 2. Build adjacency bitmasks
    adj = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Greedy Coloring of the graph
    # We want to maximize the bound, so we assign colors greedily.
    # Standard greedy coloring: sort vertices, assign smallest possible color.
    # For clique finding, we often want the "color" to be an upper bound.
    # If we color such that non-adjacent vertices share colors, the max color is an upper bound on clique size.
    # To get a tight bound, we should order vertices by degree (descending) before coloring?
    # Actually, the Tomita algorithm usually orders vertices by degree, then colors.
    
    # Let's use a simple greedy coloring.
    # Order vertices by descending degree.
    order = sorted(range(m), key=lambda i: bin(adj[i]).count('1'), reverse=True)
    
    color = [0] * m
    for v in order:
        # Find available colors
        used = set()
        # Check neighbors
        neighbors_mask = adj[v]
        # Iterate through neighbors? That's slow.
        # Better: iterate through colors and check if any neighbor has that color.
        # Since max degree is < 127, and colors are small, we can just check.
        c = 0
        while True:
            conflict = False
            # Check if any neighbor has color c
            # We can precompute a list of vertices by color? No, dynamic.
            # Simple check:
            # Is there a u in neighbors(v) such that color[u] == c?
            # We can maintain a bitmask of neighbors for each color?
            # Let's just iterate neighbors.
            u = 0
            mask = neighbors_mask
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                u_idx = lsb.bit_length() - 1
                if color[u_idx] == c:
                    conflict = True
                    break
                mask ^= lsb
            if not conflict:
                color[v] = c
                break
            c += 1
            
    # 4. Bron-Kerbosch with Pivot and Coloring
    best_clique = {idx_255}
    best_size = 1
    
    # Initial P is neighbors of 255
    P = adj[idx_255]
    X = 0
    
    # We need to iterate vertices in P sorted by color.
    # To make this efficient, we can pre-sort the global list of vertices by color.
    # But P changes.
    
    def bk(R_mask, P_mask, X_mask):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if bin(R_mask).count('1') > best_size:
                best_size = bin(R_mask).count('1')
                best_clique = set(i for i in range(m) if (R_mask >> i) & 1)
            return
            
        # Get vertices in P
        # We want to process them in order of decreasing color.
        # Extract indices
        p_list = [i for i in range(m) if (P_mask >> i) & 1]
        
        # Sort by color descending
        p_list.sort(key=lambda i: color[i], reverse=True)
        
        # Upper bound check: if best possible size <= current best, prune
        # The max color in P is an upper bound on the number of vertices we can add.
        if not p_list:
            return
            
        max_color_in_P = color[p_list[0]]
        
        if bin(R_mask).count('1') + max_color_in_P <= best_size:
            return
            
        # Pivot selection (optional optimization, often skipped in colored BK as coloring does the work)
        # Let's stick to just coloring for simplicity and robustness.
        
        for v in p_list:
            # If even adding all vertices with color >= color[v] doesn't beat best, break?
            # The loop structure handles this: we check the bound at the start of the recursion.
            # But we can also prune here: if current R size + color[v] <= best_size, break.
            # Because all subsequent vertices have color <= color[v].
            if bin(R_mask).count('1') + color[v] <= best_size:
                break
                
            # Recurse
            # New P = P & adj[v]
            # New X = X & adj[v]
            # New R = R | (1 << v)
            
            bk(R_mask | (1 << v), P_mask & adj[v], X_mask & adj[v])
            
            # Update P and X
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
            
            # Check time
            if time.time() > deadline:
                return

    bk(1 << idx_255, P, X)
    
    # Convert best_clique back to values
    result = sorted([candidates[i] for i in best_clique])
    
    # Verify
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert gcd(result[i], result[j]) > 1
            
    return result
