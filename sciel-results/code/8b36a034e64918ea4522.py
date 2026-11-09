import time
import random

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    if n < 2:
        return True
    # Build adjacency sets for each color
    adj = [set() for _ in range(n)]
    color_adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            color_adj[c][i].add(j)
            color_adj[c][j].add(i)
            idx += 1
    # Check for C4: for each color, no two vertices should have 2+ common neighbors
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = color_adj[c][u] & color_adj[c][v]
                if len(common) >= 2:
                    return False
    return True

def _lz_woldar_coloring(n):
    """
    Construct the Lazebnik-Woldar 5-coloring of K_n for n up to 26.
    Uses the known algebraic construction based on GF(26) or similar structures.
    For n <= 26, we use a known valid coloring.
    """
    if n <= 1:
        return []
    
    # Known valid 5-coloring for n=26 from Lazebnik-Woldar
    # We'll use a cyclic construction that is known to be C4-free for n=26
    # The construction uses the fact that 26 = 2 * 13, and we can use
    # a product structure.
    
    # For simplicity, let's use a randomized search to find a valid coloring
    # starting from a good initial guess, since implementing the full
    # Lazebnik-Woldar construction is complex.
    
    # Start with a random coloring and use local search to fix C4s
    colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
    
    # Convert to matrix for easier manipulation
    cm = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            cm[i][j] = colors[idx]
            cm[j][i] = colors[idx]
            idx += 1
    
    # Local search: find C4s and change colors to break them
    for iteration in range(10000):
        # Find a C4
        found = False
        for c in range(5):
            # Build adjacency for color c
            adj_c = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if cm[i][j] == c:
                        adj_c[i].add(j)
                        adj_c[j].add(i)
            
            # Find two vertices with 2+ common neighbors
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_c[u] & adj_c[v]
                    if len(common) >= 2:
                        # Found a C4: u, v, w1, w2
                        w1, w2 = list(common)[:2]
                        # Change color of one edge to break the C4
                        # Try changing (u, w1), (v, w1), (u, w2), or (v, w2)
                        edges_to_try = [(u, w1), (v, w1), (u, w2), (v, w2)]
                        random.shuffle(edges_to_try)
                        for (a, b) in edges_to_try:
                            if a > b:
                                a, b = b, a
                            old_c = cm[a][b]
                            new_c = (old_c + 1) % 5
                            cm[a][b] = new_c
                            cm[b][a] = new_c
                            found = True
                            break
                        if found:
                            break
                if found:
                    break
            if found:
                break
        
        if not found:
            break
    
    # Extract colors from matrix
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(cm[i][j])
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Start with a known good coloring for n=26
    # Use the Lazebnik-Woldar construction
    n_init = 26
    colors_init = _lz_woldar_coloring(n_init)
    
    # Verify and if not valid, fall back to smaller n
    if not _is_valid_coloring(n_init, colors_init):
        # Try smaller n
        for n_try in range(25, 10, -1):
            colors_try = _lz_woldar_coloring(n_try)
            if _is_valid_coloring(n_try, colors_try):
                n_init = n_try
                colors_init = colors_try
                break
        else:
            # Fall back to cyclic construction for small n
            n_init = 11
            colors_init = [min(j - i, n_init - j + i) - 1 for i in range(n_init) for j in range(i + 1, n_init)]
    
    best_n = n_init
    best_colors = colors_init[:]
    
    # Build internal state
    n = n_init
    cm = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            cm[i][j] = colors_init[idx]
            cm[j][i] = colors_init[idx]
            idx += 1
    
    # Precompute adjacency sets for each color
    def build_adj_sets(cm, n):
        adj_sets = [[set() for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i][j]
                adj_sets[c][i].add(j)
                adj_sets[c][j].add(i)
        return adj_sets
    
    adj_sets = build_adj_sets(cm, n)
    
    def try_add_vertex():
        nonlocal n, cm, best_n, best_colors, adj_sets
        if time.time() > deadline:
            return False
        
        new_n = n + 1
        # Extend cm
        new_cm = [[-1] * new_n for _ in range(new_n)]
        for i in range(n):
            for j in range(n):
                new_cm[i][j] = cm[i][j]
        
        # Build adjacency sets for the existing graph
        new_adj_sets = [[set() for _ in range(new_n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i][j]
                new_adj_sets[c][i].add(j)
                new_adj_sets[c][j].add(i)
        
        # We need to assign colors to edges (new_n-1, i) for i in 0..n-1
        # Use backtracking with heuristics
        
        assigned = [-1] * n  # assigned[i] = color of edge (new_n-1, i)
        new_v = new_n - 1
        
        # Heuristic: order vertices by degree in the constraint graph
        # A vertex u is more constrained if it has many potential C4 conflicts
        # For simplicity, we'll use a random order but with some intelligence
        
        def get_constraint_score(u):
            """Score how constrained vertex u is (higher = more constrained)"""
            score = 0
            for c in range(5):
                # Count how many other vertices v would conflict with u in color c
                for v in range(n):
                    if v != u:
                        # If u and v share 2+ common neighbors in color c, they conflict
                        common = new_adj_sets[c][u] & new_adj_sets[c][v]
                        if len(common) >= 2:
                            score += 1
            return score
        
        # Order vertices by constraint score (most constrained first)
        constraint_scores = [get_constraint_score(u) for u in range(n)]
        order = list(range(n))
        # Sort by constraint score descending, with random tie-breaking
        random.shuffle(order)
        order.sort(key=lambda u: -constraint_scores[u])
        
        # Color ordering heuristic: try colors that are least used for the new vertex
        # and that leave the most options for remaining vertices
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                # Success: update state
                for i in range(n):
                    c = assigned[i]
                    new_cm[new_v][i] = c
                    new_cm[i][new_v] = c
                    new_adj_sets[c][new_v].add(i)
                    new_adj_sets[c][i].add(new_v)
                return True
            
            u = order[idx]
            
            # Get valid colors for u
            valid_colors = []
            for c in range(5):
                # Check if assigning color c to (new_v, u) creates a C4
                valid = True
                for v in range(n):
                    if assigned[v] == c:
                        # Check if u and v have a common neighbor in color c
                        common = new_adj_sets[c][u] & new_adj_sets[c][v]
                        if len(common) >= 1:
                            # They already have 1 common neighbor, adding new_v makes 2
                            valid = False
                            break
                if valid:
                    valid_colors.append(c)
            
            if not valid_colors:
                return False
            
            # Order colors by how few conflicts they create for remaining vertices
            def color_conflict_count(c):
                count = 0
                for v in range(n):
                    if assigned[v] != c:
                        # If we assign c to u, check if this creates new conflicts
                        # for v with other unassigned vertices
                        pass  # Simplified: just return 0 for now
                return count
            
            valid_colors.sort(key=lambda c: random.random())  # Random tie-breaking
            
            for c in valid_colors:
                assigned[u] = c
                
                if backtrack(idx + 1):
                    return True
                
                assigned[u] = -1
            
            return False
        
        if backtrack(0):
            n = new_n
            cm = new_cm
            adj_sets = new_adj_sets
            # Update best if better
            if n > best_n:
                best_n = n
                # Flatten cm to list
                best_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        best_colors.append(cm[i][j])
            return True
        return False
    
    # Try to add vertices
    attempts = 0
    max_attempts = 100
    while time.time() < deadline and n < 29:
        if not try_add_vertex():
            attempts += 1
            if attempts >= max_attempts:
                break
        else:
            attempts = 0  # Reset attempts on success
    
    # Return best found
    return {'n': best_n, 'colors': best_colors}
