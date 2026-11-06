import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Check if we have time to do anything
    if time.time() - start_time > deadline - 1.0:
        # Fallback to a small valid graph if somehow we are late (should not happen)
        return {'n': 30, 'edges': [[0,1],[0,2],[1,2]]*10} # Incorrect fallback, just placeholder

    q = 7
    target_n = 43
    
    # Generate points of PG(2, q)
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n_total = len(points)
    
    # Compute adjacency
    adj = [[False] * n_total for _ in range(n_total)]
    
    for i in range(n_total):
        a, b, c = points[i]
        for j in range(i + 1, n_total):
            d, e, f = points[j]
            val = (c * d + b * e + a * f) % q
            if val == 0:
                adj[i][j] = True
                adj[j][i] = True
                
    # Select first target_n vertices
    selected_indices = list(range(target_n))
    old_to_new = {old: new for new, old in enumerate(selected_indices)}
    
    edges = []
    for old_i in selected_indices:
        new_i = old_to_new[old_i]
        for old_j in selected_indices:
            if old_i < old_j:
                if adj[old_i][old_j]:
                    new_j = old_to_new[old_j]
                    edges.append([new_i, new_j])
                    
    return {'n': target_n, 'edges': edges}
