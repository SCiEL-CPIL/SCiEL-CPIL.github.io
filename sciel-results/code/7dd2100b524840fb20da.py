def construct():
    import random
    import time
    import math

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_triangle_free_bitmask(adj):
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    if adj[i] & adj[j]:
                        return False
        return True

    def is_3_colorable_bitmask(adj):
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            forbidden = 0
            m = adj[v]
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if color[u] != -1:
                    forbidden |= (1 << color[u])
                m -= lsb
                
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False

        return backtrack(0)

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    # Initialize with Grötzsch Graph (Vertices 0-10)
    groetzsch_adj = [0] * n
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    groetzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,0), (5,1), (6,1), (6,2), (7,2), (7,3), (8,3), (8,4), (9,4), (9,0),
        (10,5), (10,6), (10,7), (10,8), (10,9)
    ]
    for u, v in groetzsch_edges:
        add_edge(groetzsch_adj, u, v)
        
    current_adj = groetzsch_adj[:]
    current_edges = count_edges(current_adj)
    current_is_3c = is_3_colorable_bitmask(current_adj) # Should be False
    
    best_adj = current_adj[:]
    best_edges = current_edges if not current_is_3c else 0

    start_time = time.time()
    time_limit = 115
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() - start_time < time_limit and T > T_min:
        new_adj = current_adj[:]
        move_type = random.choice(['add', 'remove', 'swap'])
        
        moved = False
        if move_type == 'add':
            non_edges = []
            for u, v in all_edges:
                if not ((current_adj[u] >> v) & 1):
                    non_edges.append((u, v))
            if non_edges:
                u, v = random.choice(non_edges)
                if not (current_adj[u] & current_adj[v]):
                    add_edge(new_adj, u, v)
                    moved = True
        elif move_type == 'remove':
            edges_in_graph = []
            for u, v in all_edges:
                if (current_adj[u] >> v) & 1:
                    edges_in_graph.append((u, v))
            if edges_in_graph:
                u, v = random.choice(edges_in_graph)
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                moved = True
        else: # swap
            edges_in_graph = []
            for u, v in all_edges:
                if (current_adj[u] >> v) & 1:
                    edges_in_graph.append((u, v))
            non_edges = []
            for u, v in all_edges:
                if not ((current_adj[u] >> v) & 1):
                    non_edges.append((u, v))
            if edges_in_graph and non_edges:
                u1, v1 = random.choice(edges_in_graph)
                u2, v2 = random.choice(non_edges)
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                if not (new_adj[u2] & new_adj[v2]):
                    add_edge(new_adj, u2, v2)
                    moved = True
        
        if not moved:
            continue
            
        new_edges = count_edges(new_adj)
        
        # Check 3-colorability
        # Optimization: If move was 'add' and current was not 3-colorable, new is not 3-colorable.
        if move_type == 'add' and not current_is_3c:
            new_is_3c = False
        else:
            new_is_3c = is_3_colorable_bitmask(new_adj)
            
        lambda_pen = 100
        old_score = current_edges - (lambda_pen if current_is_3c else 0)
        new_score = new_edges - (lambda_pen if new_is_3c else 0)
        
        delta = new_score - old_score
        
        if delta > 0:
            current_adj = new_adj
            current_edges = new_edges
            current_is_3c = new_is_3c
            if not new_is_3c and new_edges > best_edges:
                best_adj = new_adj[:]
                best_edges = new_edges
        else:
            if random.random() < math.exp(delta / T):
                current_adj = new_adj
                current_edges = new_edges
                current_is_3c = new_is_3c
                if not new_is_3c and new_edges > best_edges:
                    best_adj = new_adj[:]
                    best_edges = new_edges

        T *= cooling
        
    if best_edges == 0:
        # Fallback
        return [list(e) for e in groetzsch_edges]
        
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
