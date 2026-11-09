"""Combined program: Deterministic seed + Local Search Extension with improved heuristics."""
import time
import random
import copy
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 112.0  # Safety margin for overhead/return
    
    # 1. Seed: Greenwood-Gleason colouring of K16 (Clebsch) doubled to n=32
    # GF(16) multiplication
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Precompute log table for GF(16)
    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Initial n=32 construction
    def get_seed_graph(n):
        graph = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                graph[i][j] = c
                graph[j][i] = c
        return graph

    current_n = 32
    graph = get_seed_graph(current_n)
    best_n = current_n
    best_graph = copy.deepcopy(graph)

    def is_valid_graph(g, n):
        """Check if the graph has no monochromatic triangles."""
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = g[i][j]
                for k in range(j + 1, n):
                    if g[i][k] == c_ij and g[j][k] == c_ij:
                        return False
        return True

    def add_vertex_backtrack(g, n):
        """
        Attempts to add a new vertex at index n to graph g (size n).
        Returns True if successful (g is updated to size n+1), False otherwise.
        Uses backtracking with pruning and heuristic ordering.
        """
        # We will assign colors to edges (i, n) for i in 0..n-1
        # assignments[i] will be the color of edge (i, n)
        assignments = [-1] * n
        
        # Precompute neighbor sets for heuristic ordering
        # For each vertex i, we want to order neighbors j by how constrained they are
        # A neighbor j is more constrained if it has fewer available colors for edge (j, n)
        # given current assignments of (k, n) for k < j.
        
        # We'll use a simple heuristic: order vertices by degree in the "constraint graph"
        # or just by index. For now, we'll use a randomized order but with a bias.
        
        def backtrack(i):
            if i == n:
                return True
            
            # Get vertices k < i such that g[k][i] is fixed
            # We need to choose c for edge (i, n) such that for all k < i,
            # if g[k][i] == c, then assignments[k] != c.
            
            # Heuristic: Try colors that are "less used" in the neighborhood of i?
            # Or try colors that appear less often in g[k][i] for k < i?
            # Let's try colors 0,1,2,3 in a random order, but prioritize those
            # that don't conflict with many already-assigned neighbors.
            
            # Count conflicts for each color
            conflict_counts = [0, 0, 0, 0]
            for k in range(i):
                c_ki = g[k][i]
                if assignments[k] != -1:
                    # If assignments[k] is c, then we cannot use color c for (i, n)
                    # if g[k][i] == c (because then triangle k-i-n would be monochromatic c)
                    # Wait, the condition is: if g[k][i] == c_new and assignments[k] == c_new, then conflict.
                    # So for a candidate color c, it conflicts with k if g[k][i] == c and assignments[k] == c.
                    pass # This logic is per-candidate, so let's just count how many k have g[k][i] == c and assignments[k] == c?
                    # No, we don't know c yet. Let's just count for each color c, how many k have assignments[k] == c and g[k][i] == c.
                    # If we pick color c for (i, n), we conflict if there exists k < i with g[k][i] == c and assignments[k] == c.
                    # So for each color c, let bad_c = count of k < i such that assignments[k] == c and g[k][i] == c.
                    # If bad_c > 0, then color c is invalid.
                    # So we can precompute which colors are invalid.
                    
            invalid_colors = [False, False, False, False]
            for k in range(i):
                if assignments[k] != -1:
                    c_k = assignments[k]
                    if g[k][i] == c_k:
                        invalid_colors[c_k] = True
            
            valid_colors = [c for c in range(4) if not invalid_colors[c]]
            
            if not valid_colors:
                return False
            
            # Shuffle valid colors
            random.shuffle(valid_colors)
            
            for c in valid_colors:
                assignments[i] = c
                if backtrack(i + 1):
                    return True
            assignments[i] = -1
            return False

        if backtrack(0):
            # Success: Update graph
            # Extend graph to n+1
            for i in range(n):
                row = g[i]
                row.append(assignments[i])
            g.append([0] * (n + 1))
            last_row = g[n]
            for i in range(n):
                last_row[i] = assignments[i]
            last_row[n] = 0 # self loop, irrelevant
            return True
        return False

    attempts = 0
    max_attempts = 50000
    consecutive_failures = 0
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try to add a vertex
        if add_vertex_backtrack(graph, current_n):
            current_n += 1
            best_n = current_n
            best_graph = copy.deepcopy(graph)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            
            # If we fail to add a vertex, we need to modify the existing graph to create "room".
            # Strategy: Randomly flip some edges (Kick).
            # We must ensure the graph remains valid after the kick.
            
            # Probability of kicking: Higher if we've failed many times in a row.
            kick_prob = min(0.8, 0.05 * consecutive_failures + 0.1)
            
            if random.random() < kick_prob:
                # Save state
                old_graph = copy.deepcopy(graph)
                old_n = current_n
                
                # Perform Kick: Flip K edges
                # K should be small, e.g., 5-10% of edges
                num_edges = current_n * (current_n - 1) // 2
                # Try varying kick sizes
                if consecutive_failures < 10:
                    num_flips = random.randint(1, 5)
                elif consecutive_failures < 50:
                    num_flips = random.randint(5, 15)
                else:
                    num_flips = random.randint(10, 30)
                
                for _ in range(num_flips):
                    i = random.randint(0, current_n - 1)
                    j = random.randint(0, current_n - 1)
                    if i != j:
                        graph[i][j] = random.randint(0, 3)
                        graph[j][i] = graph[i][j]
                
                # Validate
                if is_valid_graph(graph, current_n):
                    # Kick was successful in maintaining validity.
                    pass
                else:
                    # Kick broke validity. Rollback.
                    graph = old_graph
                    current_n = old_n

    # Finalize
    n = best_n
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(best_graph[i][j])
            
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
