import numpy as np
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    q = k
    # Calculate d based on girth g
    # For g=10, d=6. For g=8, d=4?
    # Formula: d = g - 4 if (g-4)%2==0 else g-5?
    # Current: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # g=10: 10-5=5 (odd) -> d=10-4=6.
    # g=8: 8-5=3 (odd) -> d=8-4=4.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # Convert rule to arrays for faster access
    rule_a = np.zeros(d+1, dtype=np.int64)
    rule_b = np.zeros(d+1, dtype=np.int64)
    for c in range(2, d+1):
        rule_a[c], rule_b[c] = rule[c]

    # GF(4) multiplication table
    if q == 4:
        gf4_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        gf4_table = None
    
    # BFS
    start_vertex = (0, (0,) * (d + 1))
    visited = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    vertex_idx = 1
    # Use a list for the queue to avoid iterator overhead if possible, 
    # but iterating over a growing list is standard in Python BFS.
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        
        side, x_tuple = vertex
        x = np.array(x_tuple, dtype=np.int64)
        
        # Generate neighbors
        Y = np.zeros((q, d + 1), dtype=np.int64)
        Y[:, 1] = np.arange(q)
        
        for c in range(2, d + 1):
            a = rule_a[c]
            b = rule_b[c]
            if side == 0:
                if q == 4:
                    prod = gf4_table[Y[:, a], x[b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (Y[:, a] * x[b]) % q
                    Y[:, c] = (prod + x[c]) % q
            else:
                if q == 4:
                    prod = gf4_table[x[a], Y[:, b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (x[a] * Y[:, b]) % q
                    Y[:, c] = (x[c] - prod) % q
        
        for j in range(q):
            neighbor = (1 - side, tuple(Y[j]))
            
            if neighbor not in visited:
                visited[neighbor] = vertex_idx
                order.append(neighbor)
                vertex_idx += 1
            
            if side == 0:
                edges.append((visited[vertex], visited[neighbor]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
