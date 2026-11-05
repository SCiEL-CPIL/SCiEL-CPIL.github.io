import random
import time
import math

def construct():
    n = 26
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Pre-compute for each edge, the list of edges that form a C4 with it?
    # Too complex.
    # Just optimize the inner loop.
    
    # Use a 2D array for adjacency per color?
    # Or just use bitsets?
    # Python ints are arbitrary precision. We can use bitsets for neighbors.
    
    def calculate_cost(colors):
        # adj[c] is a list of n integers, each is a bitmask of neighbors
        adj = [[0] * n for _ in range(5)]
        for k in range(num_edges):
            c = colors[k]
            i, j = edges[k]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        cost = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    # Common neighbors = popcount(adj[u] & adj[v])
                    common_mask = adj[c][u] & adj[c][v]
                    # Count bits
                    common_count = bin(common_mask).count('1')
                    if common_count >= 2:
                        cost += common_count * (common_count - 1) // 2
        return cost

    colors = [random.randint(0, 4) for _ in range(num_edges)]
    current_cost = calculate_cost(colors)
    
    start_time = time.time()
    t = 50.0
    cooling = 0.9995
    
    while current_cost > 0 and time.time() - start_time < 110:
        k = random.randint(0, num_edges - 1)
        old_color = colors[k]
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        colors[k] = new_color
        new_cost = calculate_cost(colors)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / t):
            current_cost = new_cost
        else:
            colors[k] = old_color
            
        t *= cooling
        if t < 0.5:
            t = 0.5
            
    if current_cost == 0:
        return {'n': n, 'colors': colors}
        
    # Fallback: Try n=25, 24...
    for n_try in range(25, 10, -1):
        edges_try = [(i, j) for i in range(n_try) for j in range(i + 1, n_try)]
        num_edges_try = len(edges_try)
        colors_try = [random.randint(0, 4) for _ in range(num_edges_try)]
        
        def calc_cost_try(colors):
            adj = [[0] * n_try for _ in range(5)]
            for k in range(num_edges_try):
                c = colors[k]
                i, j = edges_try[k]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
            cost = 0
            for c in range(5):
                for u in range(n_try):
                    for v in range(u + 1, n_try):
                        common_mask = adj[c][u] & adj[c][v]
                        common_count = bin(common_mask).count('1')
                        if common_count >= 2:
                            cost += common_count * (common_count - 1) // 2
            return cost

        curr = calc_cost_try(colors_try)
        # Quick SA for smaller n
        t = 10.0
        limit = time.time() + 5 # 5 seconds per n
        while curr > 0 and time.time() < limit:
            k = random.randint(0, num_edges_try - 1)
            old = colors_try[k]
            new = random.randint(0, 4)
            if new == old: continue
            colors_try[k] = new
            nc = calc_cost_try(colors_try)
            d = nc - curr
            if d < 0 or random.random() < math.exp(-d / t):
                curr = nc
            else:
                colors_try[k] = old
            t *= 0.99
            if t < 0.1: t = 0.1
            
        if curr == 0:
            return {'n': n_try, 'colors': colors_try}
            
    return {'n': 1, 'colors': [0]}
