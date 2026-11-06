import time
from math import gcd

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We aim to finish much earlier.
    # If we don't finish, we return the best found so far.
    deadline = start_time + 85.0
    
    N = 512
    LIMIT = N
    
    # 1. Build adjacency masks
    # adj[v] is a bitmask of all u such that lcm(v, u) <= LIMIT
    # We use 1-based indexing for values, so bit i corresponds to value i.
    # Mask will have bits 1..N set.
    adj = [0] * (N + 1)
    
    # Precompute gcds or just compute on the fly? 
    # N is 512, so N^2 is ~260k operations. This is very fast.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            # lcm(a, b) = a * b // g
            # Check if a * b <= LIMIT * g to avoid large integers if needed, 
            # but a*b is at most 512*512 = 262144, which is small.
            if (a * b) // g <= LIMIT:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    
    # Precompute degree for each vertex
    degree = [0] * (N + 1)
    for v in range(1, N + 1):
        degree[v] = adj[v].bit_count()
    
    # Initial best clique: Find a large clique greedily to set a good lower bound
    # Greedy: Pick vertex with highest degree, add to clique, remove non-neighbors.
    def greedy_initial():
        remaining = (1 << (N + 1)) - 2  # Bits 1..N set
        clique = []
        # Order vertices by degree descending initially? 
        # Or just pick max degree in remaining at each step.
        while remaining:
            best_v = 0
            best_deg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == 0:
                break
            clique.append(best_v)
            # Next candidates must be neighbors of best_v and in remaining
            remaining &= adj[best_v]
            remaining &= ~(1 << best_v)
        return clique
    
    best_clique = greedy_initial()
    best_size = len(best_clique)
    
    # Tomita's Algorithm for Maximum Clique
    # P: candidates (bitmask)
    # X: excluded (bitmask) - not strictly needed for size, but for correctness of "maximal" vs "maximum" search path, 
    # in Tomita we usually just recurse on P and P becomes P & adj[v].
    
    # Use lists for color assignment for speed
    color_assign = [0] * (N + 1)
    
    def expand(P, path):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(path) > best_size:
                best_size = len(path)
                best_clique = list(path)
            return
            
        # Extract vertices in P
        verts = []
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            v = lsb.bit_length() - 1
            verts.append(v)
            p_temp ^= lsb
            
        if not verts:
            return
            
        # Greedy Coloring
        # We assign colors to vertices in verts.
        # color[v] will store the color index.
        # We want to minimize the number of colors used, which gives a tight upper bound.
        # Order verts by degree in P descending to potentially get tighter bounds?
        # Tomita's original paper suggests ordering by degree.
        degs = []
        for v in verts:
            degs.append((adj[v] & P).bit_count())
        
        # Sort vertices by degree descending
        verts_sorted = [v for _, v in sorted(zip(degs, verts), key=lambda x: -x[0])]
        
        max_color = 0
        colored_mask = 0
        
        for v in verts_sorted:
            # Find available colors
            # A color is blocked if a neighbor in colored_mask has that color
            blocked = 0
            neighbors_colored = adj[v] & colored_mask
            nc = neighbors_colored
            while nc:
                lsb = nc & -nc
                u = lsb.bit_length() - 1
                # If u is colored, block its color
                blocked |= (1 << color_assign[u])
                nc ^= lsb
                
            # Find smallest available color
            c = 0
            while blocked & (1 << c):
                c += 1
            color_assign[v] = c
            colored_mask |= (1 << v)
            if c > max_color:
                max_color = c
                
        # Upper bound: max_color + 1
        # If current path size + max possible addition <= best_size, prune
        if len(path) + max_color + 1 <= best_size:
            return

        # Branching
        # We process vertices in descending order of their color.
        # This ensures we explore the most promising branches first.
        
        # Create a list of (color, vertex) pairs
        colored_verts = [(color_assign[v], v) for v in verts_sorted]
        # Sort by color descending
        colored_verts.sort(key=lambda x: -x[0])
        
        local_P = P
        for c, v in colored_verts:
            if time.time() > deadline:
                return
            
            # Check if v is still in local_P
            if (local_P >> v) & 1 == 0:
                continue
                
            # Pruning: if len(path) + c + 1 <= best_size, we can break?
            # The bound for the remaining vertices (with color <= c) is c + 1.
            # If len(path) + c + 1 <= best_size, no further branch can improve.
            if len(path) + c + 1 <= best_size:
                break
                
            # Recurse
            new_P = local_P & adj[v]
            expand(new_P, path + [v])
            
            # Remove v from local_P
            local_P &= ~(1 << v)
    
    # Initial P is all vertices 1..N
    P = 0
    for i in range(1, N + 1):
        P |= (1 << i)
        
    expand(P, [])
    
    # Fallback
    if not best_clique:
        best_clique = [1]
        
    return best_clique
