"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Build the 32-vertex Greenwood-Gleason coloring
    n_start = 32
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    def adj_from_colors(n, colors):
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i, j] = c
                adj[j, i] = c
                idx += 1
        return adj
    
    def colors_from_adj(adj):
        n = adj.shape[0]
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(adj[i, j]))
        return colors
    
    def try_add_vertex(adj, max_backtracks=500000):
        """Try to add a new vertex to the coloring. Returns new adj or None."""
        n = adj.shape[0]
        
        # For each color c, build the graph G_c on existing vertices
        # G_c has edge (i,j) if adj[i,j] == c
        # We need to assign each existing vertex i a color c_i for edge (new, i)
        # such that for each c, the set {i : c_i == c} is independent in G_c
        G = {}
        for c in range(4):
            G[c] = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == c:
                        G[c][i].add(j)
                        G[c][j].add(i)
        
        assignment = [-1] * n
        
        def is_consistent(pos, c):
            for j in range(pos):
                if assignment[j] == c and j in G[c][pos]:
                    return False
            return True
        
        def backtrack(pos, backtrack_count):
            if pos == n:
                return True
            if backtrack_count[0] > max_backtracks:
                return False
            
            color_order = [0, 1, 2, 3]
            random.shuffle(color_order)
            
            for c in color_order:
                if is_consistent(pos, c):
                    assignment[pos] = c
                    backtrack_count[0] += 1
                    if backtrack(pos + 1, backtrack_count):
                        return True
                    assignment[pos] = -1
            
            return False
        
        backtrack_count = [0]
        if backtrack(0, backtrack_count):
            new_adj = np.zeros((n + 1, n + 1), dtype=np.int8)
            new_adj[:n, :n] = adj
            for i in range(n):
                new_adj[n, i] = assignment[i]
                new_adj[i, n] = assignment[i]
            return new_adj
        
        return None
    
    adj = adj_from_colors(n_start, colors)
    best_adj = adj.copy()
    best_n = n_start
    
    while time.time() < deadline - 10:
        new_adj = try_add_vertex(adj)
        if new_adj is None:
            break
        adj = new_adj
        best_adj = adj.copy()
        best_n = adj.shape[0]
    
    best_colors = colors_from_adj(best_adj)
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
