import math
import time

def construct():
    n = 256
    start_time = time.time()
    time_limit = 80.0

    # Precompute coprime relationships using bitmasks
    # adj_mask[i] is a bitmask where bit j is set if gcd(i,j) == 1
    adj_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if math.gcd(i, j) == 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    # Vertices sorted by weight descending
    vertices = list(range(n, 0, -1))

    best_weight = 0
    best_clique = []

    def weighted_greedy_color(candidates):
        """
        Weighted greedy coloring bound for maximum weight clique.
        candidates is a list of vertices (integers).
        Returns an upper bound on the max weight clique in the induced subgraph.
        
        We color the graph such that adjacent vertices get different colors.
        A clique can have at most one vertex per color.
        Bound = sum of max weight vertex in each color class.
        
        For tighter bounds, process higher-weight vertices first.
        """
        if not candidates:
            return 0

        # Sort candidates by weight descending for tighter coloring
        # (candidates may already be sorted, but ensure it)
        candidates_sorted = sorted(candidates, reverse=True)
        
        # Color classes: list of lists of vertices
        color_classes = []
        
        for v in candidates_sorted:
            # Find a color where v is NOT adjacent to any vertex in that color class
            # In our graph, edge = coprime, so adjacent means gcd==1
            assigned = False
            for k, color_verts in enumerate(color_classes):
                # Check if v is adjacent (coprime) to any vertex in this color class
                conflict = False
                for u in color_verts:
                    if adj_mask[v] & (1 << u):
                        conflict = True
                        break
                if not conflict:
                    color_classes[k].append(v)
                    assigned = True
                    break
            
            if not assigned:
                color_classes.append([v])
        
        # Bound: sum of max weight in each color class
        # Since we process in descending order, the first vertex added to each class is the max
        bound = 0
        for color_verts in color_classes:
            bound += color_verts[0]
            
        return bound

    def expand(candidates, current_weight, current_clique):
        nonlocal best_weight, best_clique
        
        if time.time() - start_time > time_limit:
            return
            
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_clique[:]
            return
        
        # Calculate upper bound using weighted greedy coloring
        ub = weighted_greedy_color(candidates)
        if current_weight + ub <= best_weight:
            return
        
        # Pick the highest-weight vertex to branch on
        v = candidates[0]
        
        # Branch 1: Include v
        # New candidates must be coprime with v (neighbors in our graph)
        new_candidates = [u for u in candidates[1:] if adj_mask[v] & (1 << u)]
        expand(new_candidates, current_weight + v, current_clique + [v])
        
        # Branch 2: Exclude v
        expand(candidates[1:], current_weight, current_clique)

    # Start the search
    expand(vertices, 0, [])
    
    # Ensure we return a valid list of Python ints
    return list(best_clique)

# EVOLVE-BLOCK-END
