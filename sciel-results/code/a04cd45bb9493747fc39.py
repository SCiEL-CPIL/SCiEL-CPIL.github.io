import random
import sys
import time

def construct():
    """
    Greedy randomized construction for 5-colourings of K_n without monochromatic C4.
    Uses a heuristic to minimize the creation of "forbidden" pairs (pairs with exactly 1 common neighbor).
    """
    max_n = 29
    num_colors = 5
    
    # Try to build from largest n down
    for n in range(max_n, 25, -1):
        result = _build_coloring_optimized(n, num_colors)
        if result is not None:
            return result

    # Fallback to known construction for n=11
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

def _build_coloring_optimized(n, num_colors, max_attempts=200):
    """
    Attempt to build a valid 5-coloring of K_n without monochromatic C4.
    Returns dict if successful, else None.
    """
    start_time = time.time()
    timeout = 115  # Leave buffer for 120s limit
    
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize data structures
        # colors[i][j] = color of edge (i,j), -1 if unassigned
        colors = [[-1] * n for _ in range(n)]
        
        # common_count[c][u][v] = number of common neighbors in color c for pair (u,v), u < v
        # We use a list of 5 n x n matrices
        common_count = [[[0] * n for _ in range(n)] for _ in range(num_colors)]
        
        rng = random.Random(attempt * 12345 + 67890)
        shuffled_edges = edges[:]
        rng.shuffle(shuffled_edges)
        
        valid = True
        for (u, v) in shuffled_edges:
            # Try each color in random order, or pick best cost
            best_color = -1
            best_cost = float('inf')
            
            color_order = list(range(num_colors))
            rng.shuffle(color_order)
            
            for c in color_order:
                cost, is_valid = _evaluate_assignment(colors, common_count, n, u, v, c)
                if is_valid:
                    if cost < best_cost:
                        best_cost = cost
                        best_color = c
            
            if best_color == -1:
                valid = False
                break
            
            _apply_assignment(colors, common_count, n, u, v, best_color)
        
        if valid:
            flat_colors = [colors[i][j] for i in range(n) for j in range(i+1, n)]
            return {'n': n, 'colors': flat_colors}
    
    return None

def _evaluate_assignment(colors, common_count, n, u, v, c):
    """
    Check if assigning color c to edge (u,v) is valid.
    Calculate cost: number of new pairs that will have exactly 1 common neighbor in color c.
    """
    # Check if (u,v) already has 2 common neighbors in c
    # common_count[c][u][v] should be < 2. If it is 2, it's already a C4 (shouldn't happen).
    # If it is 1, it remains 1.
    # Adding (u,v) does not change common_count[c][u][v].
    
    cost = 0
    
    # Check neighbors of u in color c
    for w in range(n):
        if w == u or w == v:
            continue
        if colors[u][w] == c:
            # Pair (v, w) gains u as common neighbor
            min_vw = v if v < w else w
            max_vw = w if v < w else v
            new_count = common_count[c][min_vw][max_vw] + 1
            if new_count >= 2:
                return float('inf'), False
            if new_count == 1:
                cost += 1
                
    # Check neighbors of v in color c
    for w in range(n):
        if w == u or w == v:
            continue
        if colors[v][w] == c:
            # Pair (u, w) gains v as common neighbor
            min_uw = u if u < w else w
            max_uw = w if u < w else u
            new_count = common_count[c][min_uw][max_uw] + 1
            if new_count >= 2:
                return float('inf'), False
            if new_count == 1:
                cost += 1
                
    return cost, True

def _apply_assignment(colors, common_count, n, u, v, c):
    """
    Apply the assignment of color c to edge (u,v) and update common_count.
    """
    colors[u][v] = c
    colors[v][u] = c
    
    # Update common_count
    for w in range(n):
        if w == u or w == v:
            continue
        if colors[u][w] == c:
            min_vw = v if v < w else w
            max_vw = w if v < w else v
            common_count[c][min_vw][max_vw] += 1
        if colors[v][w] == c:
            min_uw = u if u < w else w
            max_uw = w if u < w else u
            common_count[c][min_uw][max_uw] += 1
