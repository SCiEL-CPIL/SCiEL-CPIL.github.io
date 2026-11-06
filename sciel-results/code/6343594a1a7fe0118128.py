"""
Structural change: Use a systematic backtracking with constraint propagation
to build a 4-coloring of K_n without monochromatic triangles, starting from
a known good base and extending greedily with lookahead.

The key insight: instead of random search for extension, use a structured
backtracking that tries to add vertices one at a time, using the constraint
that each color class must be triangle-free. We use a greedy + backtracking
hybrid with time management.

We start from the known K16 Greenwood-Gleason construction and try to
extend it systematically. The constraint for adding vertex v:
  For each color c, the set of vertices connected to v by color c must form
  an independent set in the color-c subgraph.
"""

import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill

    # --- Build base K16 Greenwood-Gleason colouring using GF(16) ---
    # GF(16) with polynomial x^4 + x + 1 (0b10011)
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

    # Base K16: colour of edge (i,j) = log[i^j] % 3
    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log[i ^ j] % 3

    # --- Build K32: two copies of K16, cross edges all colour 3 ---
    n32 = 32
    adj = [[0] * n32 for _ in range(n32)]
    for i in range(n32):
        for j in range(i + 1, n32):
            c = base16[(i % 16, j % 16)] if i // 16 == j // 16 else 3
            adj[i][j] = c
            adj[j][i] = c

    adj_np = np.array(adj, dtype=np.int8)

    # --- Fast constraint checking for adding a vertex ---
    # For a candidate color_map (color of edge from new vertex to each existing vertex),
    # check that for each color c, no two vertices in group c have a color-c edge between them.
    
    def check_color_map(adj_np, n, color_map):
        """Check if color_map is valid for a new vertex."""
        # For each color c, collect vertices assigned color c
        # Then check no pair in that group has a color-c edge
        for c in range(4):
            grp = [i for i in range(n) if color_map[i] == c]
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    if adj_np[grp[a], grp[b]] == c:
                        return False
        return True

    def try_add_vertex(adj_np, n, max_attempts, start_time, deadline):
        """Try to find a valid color_map for a new vertex."""
        # Strategy: try random assignments, but with some structure
        # First try: all same color (if valid)
        # Then: random with bias
        
        # Precompute for each pair (i,j) the color, to speed up checks
        # Actually, let's use a smarter approach:
        # For each vertex i, and each color c, the "forbidden" colors for the new vertex
        # are constrained by the neighbors of i in color c.
        
        # Let's use a constraint propagation approach:
        # For the new vertex v, let color(v,i) = c_i
        # Constraint: if color(v,i) = color(v,j) = c, then color(i,j) != c
        
        # This is equivalent to: for each c, the set {i : c_i = c} must be an independent set
        # in the graph G_c where G_c has edge (i,j) iff color(i,j) = c.
        
        # Greedy approach: assign colors one by one, maintaining consistency
        # Use backtracking if needed
        
        # For speed, let's try a few strategies:
        
        # Strategy 1: Random
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            color_map = [random.randint(0, 3) for _ in range(n)]
            if check_color_map(adj_np, n, color_map):
                return color_map
        
        # Strategy 2: Greedy with randomization
        # Assign colors to vertices in random order, choosing a color that doesn't violate constraints
        # with previously assigned vertices
        order = list(range(n))
        random.shuffle(order)
        color_map = [-1] * n
        for idx, v in enumerate(order):
            # Determine which colors are still possible for v
            # A color c is forbidden if there exists a previously assigned vertex u
            # with color_map[u] = c and adj_np[u][v] = c
            forbidden = [False] * 4
            for u in order[:idx]:
                cu = color_map[u]
                if cu >= 0 and adj_np[u][v] == cu:
                    forbidden[cu] = True
            # Pick a non-forbidden color
            available = [c for c in range(4) if not forbidden[c]]
            if not available:
                return None  # Dead end, try different order
            color_map[v] = random.choice(available)
        
        if check_color_map(adj_np, n, color_map):
            return color_map
        
        return None

    def extend_coloring(adj_np, n, max_vertices, deadline):
        """Try to extend the coloring by up to max_vertices."""
        current_adj = adj_np.copy()
        current_n = n
        start_time = time.time()
        
        for _ in range(max_vertices):
            if time.time() > deadline:
                break
            
            # Budget per vertex: use remaining time divided by remaining vertices
            remaining_time = deadline - time.time()
            if remaining_time < 1.0:
                break
            
            # Try to add one vertex
            max_attempts = int(remaining_time * 50)  # ~50k attempts per second
            max_attempts = max(1000, min(max_attempts, 100000))
            
            color_map = try_add_vertex(current_adj, current_n, max_attempts, start_time, deadline)
            
            if color_map is not None:
                # Add the vertex
                new_n = current_n + 1
                new_adj = np.zeros((new_n, new_n), dtype=np.int8)
                new_adj[:current_n, :current_n] = current_adj
                for i in range(current_n):
                    new_adj[i, new_n - 1] = color_map[i]
                    new_adj[new_n - 1, i] = color_map[i]
                current_adj = new_adj
                current_n = new_n
            else:
                break
        
        if current_n > n:
            return current_n, current_adj
        return None

    # --- Try to extend K32 ---
    best_n = 32
    best_adj = adj_np
    
    # Try extending from K32
    if time.time() < deadline - 5.0:
        result = extend_coloring(adj_np, 32, max_vertices=19, deadline=deadline)
        if result is not None:
            best_n, best_adj = result

    # --- Build final result ---
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(int(best_adj[i, j]))
    
    return {'n': best_n, 'colors': colors}
