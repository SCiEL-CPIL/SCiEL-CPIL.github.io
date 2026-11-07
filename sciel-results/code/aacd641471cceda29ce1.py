"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    import time
    from itertools import combinations
    
    deadline = time.time() + 80  # safety margin before 90s kill
    
    n = P['n']
    # Build adjacency as bitmasks for speed
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def greedy_color_order(candidates):
        """Color candidates greedily, return (colored_list, max_color) where
        colored_list is sorted by color (lowest first), giving an upper bound."""
        colored = []
        colors = {}
        # Order candidates by degree descending for better coloring
        cand_list = list(candidates)
        cand_list.sort(key=lambda x: bin(adj[x] & ((1 << n) - 1)).count('1'), reverse=True)
        
        color_idx = 0
        remaining = set(cand_list)
        while remaining:
            color_idx += 1
            # Greedily pick vertices that don't conflict with same-color set
            color_class = set()
            for v in list(remaining):
                if all(v not in color_class for _ in [1]):
                    # Check v is not adjacent to any in color_class
                    conflict = False
                    for w in color_class:
                        if (adj[v] >> w) & 1:
                            conflict = True
                            break
                    if not conflict:
                        color_class.add(v)
                        colors[v] = color_idx
            remaining -= color_class
            colored.extend(color_class)
        
        # Sort by color so that vertices with lower color come first
        colored.sort(key=lambda x: colors[x])
        return colored, color_idx
    
    def max_clique_search(candidates, current_clique):
        """Branch and bound for maximum clique.
        candidates: set of vertices that can be added
        current_clique: list of vertices in current clique
        Returns best clique found."""
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return
        
        # Get upper bound via greedy coloring
        cand_list = list(candidates)
        colored, max_color = greedy_color_order(cand_list)
        
        if len(current_clique) + max_color <= best_size:
            return
        
        # Process vertices in reverse color order (highest color first for pruning)
        for i in range(len(colored) - 1, -1, -1):
            v = colored[i]
            # Upper bound: current size + number of vertices from i onwards
            if len(current_clique) + i + 1 <= best_size:
                break
            if time.time() > deadline:
                break
            
            # Try adding v
            current_clique.append(v)
            new_candidates = candidates & (adj[v] >> 0) & ((1 << n) - 1)
            new_candidates = {x for x in candidates if (adj[v] >> x) & 1}
            max_clique_search(new_candidates, current_clique)
            current_clique.pop()
            candidates.discard(v)
        
        if len(current_clique) > best_size:
            best_size = len(current_clique)
            best_clique = list(current_clique)
    
    # Initialize
    best_clique = []
    best_size = 0
    
    # Start with all vertices
    all_vertices = set(range(n))
    max_clique_search(all_vertices, [])
    
    # Fallback: try to find a good clique greedily if branch-and-bound didn't find much
    if best_size < 2:
        # Simple greedy: pick vertex with highest degree, then common neighbors
        best = []
        for v in range(n):
            if best_size == 0 or bin(adj[v]).count('1') > bin(adj[best[0]]).count('1'):
                best = [v]
        if best:
            best_clique = best
            best_size = 1
    
    # Ensure at least 2 branches
    if best_size < 2:
        # Find any edge
        u, v = P['edges'][0]
        best_clique = [u, v]
        best_size = 2
    elif best_size == 2:
        # Verify it's actually an edge
        u, v = best_clique
        if not ((adj[u] >> v) & 1):
            u, v = P['edges'][0]
            best_clique = [u, v]
    
    # Build result: branches and paths
    branches = list(best_clique)
    paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            a, b = branches[i], branches[j]
            # Verify they're adjacent
            if (adj[a] >> b) & 1:
                paths.append([a, b])
            else:
                # Should not happen for a clique, but fallback
                paths.append([a, b])
    
    # Convert to proper Python types
    branches = [int(x) for x in branches]
    paths = [[int(x) for x in p] for p in paths]
    
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
