import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Try to find a valid graph on n vertices
    # The best known is 39. We aim for 39.
    for n in [39, 38, 37]:
        result = try_construct(n, deadline)
        if result:
            return result
            
    # Fallback to a known valid smaller graph if all else fails
    # Ten disjoint triangles: n=30, alpha=10
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def try_construct(n, deadline):
    # Bitmask representation for neighbors
    # neighbors[i] is a bitmask of neighbors of vertex i
    neighbors = [0] * n
    
    # Helper to check if adding edge (u, v) creates a C4
    # It creates a C4 if they share a common neighbor
    def can_add_edge(u, v):
        if (neighbors[u] >> v) & 1:
            return False # Already connected
        if neighbors[u] & neighbors[v]:
            return False # Common neighbor exists -> C4
        return True

    def add_edge(u, v):
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)

    # Helper to compute independence number using branch and bound
    # Uses greedy coloring for upper bound
    def max_independent_set():
        # If time is up, we might skip exact check or use a faster heuristic
        # But we need exactness for validity.
        
        # Greedy coloring to get upper bound
        # We want to color the graph such that each color class is an independent set.
        # Actually, we want to color the complement? No.
        # Standard Bron-Kerbosch with pivoting or simple branch and bound on independent set.
        
        # Let's use a simple recursive branch and bound.
        # To speed up, we use a greedy independent set to get a lower bound.
        
        def greedy_independent_set(vertices):
            # vertices is a list or bitmask
            # Greedy pick: pick vertex with min degree in induced subgraph?
            # For speed, just pick first available?
            # Better: sort by degree in induced subgraph.
            
            # For simplicity and speed in this context, we'll use a heuristic:
            # Pick vertex with lowest degree in the remaining graph.
            
            # However, implementing dynamic degree sorting is expensive.
            # Let's use a static greedy: pick vertex, remove neighbors.
            
            remaining = vertices
            count = 0
            while remaining:
                # Pick a vertex with minimum degree in the remaining subgraph
                min_deg = float('inf')
                pick = -1
                temp = remaining
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    deg = (neighbors[v] & remaining).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        pick = v
                count += 1
                remaining &= ~(neighbors[pick] | (1 << pick))
            return count
        
        # Branch and Bound
        best_size = 0
        
        def bnb(vertices, current_size):
            nonlocal best_size
            if time.time() > deadline:
                return -1 # Signal timeout
            
            if not vertices:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Upper bound: current_size + number of vertices remaining
            # If this doesn't beat best, prune
            if current_size + vertices.bit_count() <= best_size:
                return
            
            # Greedy lower bound update
            greedy_size = greedy_independent_set(vertices)
            if current_size + greedy_size > best_size:
                best_size = current_size + greedy_size
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in remaining?
            # Or just lowest index?
            # Let's pick lowest index for determinism in search path, but high degree might prune better?
            # Actually, for independent set, we branch on include/exclude.
            # Including v forces excluding N(v).
            
            # Pick v
            v = (vertices & -vertices).bit_length() - 1
            
            # Branch 1: Include v
            # New vertices = vertices without v and without N(v)
            new_vertices = vertices & ~(neighbors[v] | (1 << v))
            ret = bnb(new_vertices, current_size + 1)
            if ret == -1: return -1
            
            # Branch 2: Exclude v
            new_vertices = vertices & ~(1 << v)
            ret = bnb(new_vertices, current_size)
            if ret == -1: return -1
            
        all_vertices = (1 << n) - 1
        ret = bnb(all_vertices, 0)
        if ret == -1:
            return None # Timeout
        return best_size

    # Randomized Greedy Construction
    random.seed(int(time.time() * 1000) % 10000)
    
    for attempt in range(500):
        if time.time() > deadline:
            break
            
        neighbors = [0] * n
        edge_count = 0
        
        # 1. Greedy Random Edge Addition
        potential_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(potential_edges)
        
        for u, v in potential_edges:
            if can_add_edge(u, v):
                add_edge(u, v)
                edge_count += 1
            if time.time() > deadline:
                break
        
        # 2. Local Search to increase density and reduce alpha
        # We want to add edges that connect vertices in large independent sets.
        # We will iterate:
        # a. Check alpha. If <= 10, done.
        # b. Find a large independent set S.
        # c. Try to add edges between vertices in S that are valid.
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            alpha = max_independent_set()
            if alpha is None:
                break
            if alpha <= 10:
                # Success
                edges = []
                for i in range(n):
                    mask = neighbors[i]
                    while mask:
                        j = (mask & -mask).bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask &= mask - 1
                return {'n': n, 'edges': edges}
                
            # Find a large independent set to target
            # We can reuse the greedy independent set logic or just pick a random subset?
            # Let's find one using the greedy method used in B&B
            def find_greedy_independent_set(vertices):
                remaining = vertices
                s = set()
                while remaining:
                    min_deg = float('inf')
                    pick = -1
                    temp = remaining
                    while temp:
                        v = (temp & -temp).bit_length() - 1
                        temp &= temp - 1
                        deg = (neighbors[v] & remaining).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            pick = v
                    s.add(pick)
                    remaining &= ~(neighbors[pick] | (1 << pick))
                return s
            
            S = find_greedy_independent_set((1 << n) - 1)
            
            added = False
            S_list = list(S)
            # Shuffle to randomize edge additions
            random.shuffle(S_list)
            
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u = S_list[i]
                    v = S_list[j]
                    if can_add_edge(u, v):
                        add_edge(u, v)
                        added = True
                        # Removing one edge from independent set S reduces its size by 1
                        # We can break early or continue?
                        # Continuing might find more edges.
                        # But S is no longer independent.
                        # For simplicity, we just add all valid edges in current S.
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
                    
            if not added:
                improved = False
            else:
                # If we added edges, we might need to restart the local search loop
                # because the independent set structure has changed.
                improved = True
        
    return None
