import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target n
    target_n = 17
    
    # Seed coloring for n=9
    # colors[i][j] for i<j
    # Seed: min(j-i, n-j+i) - 1
    def get_seed_color(i, j, n=9):
        return min(j - i, n - j + i) - 1

    best_n = 9
    best_colors = [get_seed_color(i, j) for i in range(9) for j in range(i + 1, 9)]

    # Helper to check if adding edge (u, v) with color c creates a C6
    # We maintain adj lists for each color
    # adj[c] is a list of sets or bitmasks
    # Since n is small, lists of sets are fine, but bitmasks are faster for intersection.
    # Let's use bitmasks.
    
    # We will try to build K_17.
    # State:
    # edge_color[i][j] for i<j
    # adj[c][v] : bitmask of neighbors of v in color c
    
    # We fix the first 9 vertices to the seed coloring.
    # Then we backtrack on vertices 9..16.
    
    def search():
        nonlocal best_n, best_colors
        
        # Initialize with seed
        n_seed = 9
        edge_color = [[-1] * target_n for _ in range(target_n)]
        adj = [ [0] * target_n for _ in range(4) ]
        
        for i in range(n_seed):
            for j in range(i + 1, n_seed):
                c = get_seed_color(i, j)
                edge_color[i][j] = c
                edge_color[j][i] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        # Backtracking function
        # v is the new vertex to add (from n_seed to target_n - 1)
        # u is the vertex we are currently connecting v to (from v-1 down to 0)
        
        def backtrack(v, u):
            if time.time() > deadline:
                return False
            
            if v == target_n:
                # Success
                nonlocal best_n, best_colors
                cols = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        cols.append(edge_color[i][j])
                best_n = target_n
                best_colors = cols
                return True
            
            if u < 0:
                return backtrack(v + 1, v)
            
            # Try colors 0..3 for edge (v, u)
            # Heuristic: try colors that are less likely to cause conflicts?
            # Or just randomize?
            colors_to_try = [0, 1, 2, 3]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Check if assigning (v, u) = c creates a C6
                # A C6 must use edge (v, u) and another edge (v, w) of color c.
                # So we check all w < v such that edge_color[v][w] == c.
                valid = True
                # Iterate w from v-1 down to 0
                # We only need to check w that are already assigned (which are all w < v)
                # and have color c.
                
                # Optimization: Precompute list of w for each color?
                # Or just iterate.
                for w in range(v - 1, -1, -1):
                    if edge_color[v][w] == c:
                        # Check if there is a simple path of length 4 between u and w in G_c (induced by 0..v-1)
                        if has_simple_path_len_4(adj, c, u, w, v):
                            valid = False
                            break
                
                if valid:
                    # Assign
                    edge_color[v][u] = c
                    edge_color[u][v] = c
                    adj[c][v] |= (1 << u)
                    adj[c][u] |= (1 << v)
                    
                    if backtrack(v, u - 1):
                        return True
                    
                    # Unassign
                    edge_color[v][u] = -1
                    edge_color[u][v] = -1
                    adj[c][v] ^= (1 << u)
                    adj[c][u] ^= (1 << v)
            
            return False
        
        return backtrack(n_seed, n_seed - 1)
    
    # Run multiple restarts with different random seeds for color ordering
    for restart in range(100):
        if time.time() > deadline:
            break
        random.seed(restart)
        if search():
            break
            
    return {'n': best_n, 'colors': best_colors}

def has_simple_path_len_4(adj, c, u, w, v_limit):
    """
    Check if there is a simple path of length 4 from u to w in the graph of color c,
    using only vertices < v_limit.
    Path: u - a - b - d - w
    """
    # adj[c] is the adjacency bitmask for color c.
    # We need to find a, b, d such that:
    # 1. (1<<a) in adj[c][u]
    # 2. (1<<b) in adj[c][a]
    # 3. (1<<d) in adj[c][b]
    # 4. (1<<w) in adj[c][d]
    # And u, a, b, d, w are distinct.
    
    # Since v_limit <= 17, we can iterate.
    # But we can use bitmasks to speed up.
    
    # N1 = neighbors of u
    n1 = adj[c][u]
    
    # For each a in N1:
    #   N2 = neighbors of a
    #   For each b in N2 (b != u):
    #     N3 = neighbors of b
    #     For each d in N3 (d != u, d != a):
    #       if (1<<w) in N3 and w != u, w != a, w != b:
    #         return True
    
    # This is O(deg(u) * deg(a) * deg(b)).
    # With n=17, max deg is 16. 16*16*16 = 4096 ops. Very fast.
    
    n1 = adj[c][u]
    while n1:
        a_bit = n1 & -n1
        a = a_bit.bit_length() - 1
        n1 ^= a_bit
        
        n2 = adj[c][a] & ~(1 << u) # b cannot be u
        while n2:
            b_bit = n2 & -n2
            b = b_bit.bit_length() - 1
            n2 ^= b_bit
            
            n3 = adj[c][b] & ~(1 << u) & ~(1 << a) # d cannot be u or a
            # We need w to be in n3
            # And w cannot be b
            if (1 << w) & n3:
                # Check if w != b
                if w != b:
                    return True
            # Also need to ensure w != u, w != a, w != b
            # w is distinct from u, a, b by construction of n3?
            # n3 excludes u and a.
            # We explicitly check w != b.
            # w is fixed.
            
    return False
