import time
import sys

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # 1. Build adjacency list of "bad" pairs
    # bad[i][j] is True if i and j cannot be in the same set
    # i.e., (i*j) % (i+j) == 0
    bad = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                bad[i][j] = True
                bad[j][i] = True

    # 2. Greedy heuristic to find a large independent set
    # Sort by degree (number of bad neighbors) ascending
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i != j and bad[i][j]:
                degrees[i] += 1
    
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: degrees[x])
    
    best_set = set()
    blocked = [False] * (N + 1)
    for v in vertices:
        if not blocked[v]:
            best_set.add(v)
            for u in range(1, N + 1):
                if bad[v][u]:
                    blocked[u] = True
    
    best_size = len(best_set)
    
    # 3. Branch and Bound to improve
    # We will try to find an independent set of size > best_size
    # We use a recursive search.
    # State: candidates (list), current_set (set)
    # To speed up, we use a greedy coloring bound on the candidates.
    
    def greedy_color_bound(candidates):
        # Returns an upper bound on the size of the independent set in candidates
        # using greedy coloring.
        if not candidates:
            return 0
        # Sort candidates by degree in the subgraph induced by candidates
        # To keep it fast, we might just use the global degree or a simple heuristic
        # Let's use a simple greedy coloring on the subgraph
        sub_candidates = list(candidates)
        # Order by degree in subgraph
        def get_deg(v):
            d = 0
            for u in sub_candidates:
                if v != u and bad[v][u]:
                    d += 1
            return d
        
        sub_candidates.sort(key=get_deg, reverse=True)
        
        colors = {}
        max_color = 0
        # Standard greedy coloring: assign smallest available color
        for v in sub_candidates:
            used_colors = set()
            for u in sub_candidates:
                if v != u and bad[v][u] and u in colors:
                    used_colors.add(colors[u])
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
        
        # The bound is the number of vertices? No.
        # In Tomita's algorithm, the bound is derived from the color classes.
        # Specifically, if we order vertices by color, the bound for a suffix is the max color in that suffix?
        # Actually, a simpler valid upper bound is the number of vertices.
        # But we can use the fact that each color class is an independent set.
        # This doesn't give an upper bound on the size of the MAX independent set.
        # Wait. The chromatic number $\chi$ is the minimum colors.
        # $\alpha(G) \ge |V|/\chi(G)$. This is a lower bound.
        # We need an upper bound for pruning.
        # The number of vertices is a trivial upper bound.
        # A better upper bound:
        # If we greedily color, and the max color is K, it doesn't mean $\alpha(G) \le K$.
        # Example: Path of 4 vertices. $\chi=2$. $\alpha=2$.
        # Example: $K_5$. $\chi=5$. $\alpha=1$.
        # So max color is not an upper bound on $\alpha$.
        
        # However, in the context of the search tree, we can use the following:
        # We order the candidates.
        # The bound is simply the number of candidates if we don't have a good heuristic.
        # Let's try a different approach: Local Search / Simulated Annealing?
        # Or just a simple swap heuristic.
        
        # Given the time constraint and complexity of implementing a tight bound correctly from scratch,
        # and the fact that the greedy set is likely close to optimal,
        # let's use a simple "try to add" or "swap" local search.
        
        # Actually, the Tomita bound IS the number of color classes?
        # No.
        # Let's stick to a simpler, effective local search.
        return len(sub_candidates) # Trivial bound, no pruning benefit yet

    # Let's implement a simple local search to improve the greedy set
    # Try to swap one vertex for two? Or add one?
    # Since we can't add if blocked, we must swap.
    
    # Heuristic: 
    # For each vertex v in best_set, try removing it and adding any two vertices that become unblocked?
    # This is complex.
    
    # Alternative: Randomized Restart Greedy
    # Run greedy multiple times with different orderings (e.g., random tie-breaking).
    
    import random
    
    best_size_found = best_size
    best_set_found = set(best_set)
    
    for i in range(100): # 100 restarts
        if time.time() > deadline:
            break
        
        # Randomize order slightly
        verts = list(range(1, N + 1))
        random.shuffle(verts)
        # Sort by degree, but with random tie breaker
        verts.sort(key=lambda x: (degrees[x], random.random()))
        
        current_set = set()
        blocked = [False] * (N + 1)
        for v in verts:
            if not blocked[v]:
                current_set.add(v)
                for u in range(1, N + 1):
                    if bad[v][u]:
                        blocked[u] = True
        
        if len(current_set) > best_size_found:
            best_size_found = len(current_set)
            best_set_found = set(current_set)
            
    return list(best_set_found)

construct()
