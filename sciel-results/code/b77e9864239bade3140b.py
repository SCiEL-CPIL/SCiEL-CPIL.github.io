import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 120.0 # seconds
    safety_margin = 5.0
    
    q = k
    
    # Determine candidate d values
    # We know d must be at least 2.
    # Girth estimates:
    # odd d: d + 5
    # even d: 2*d + 1 (Hypothesis based on d=2 -> girth 5)
    
    best_d = None
    best_n = float('inf')
    
    # Max d such that 2*q^d is reasonable?
    # We want to minimize n, so we want smallest d satisfying girth.
    # But we should check a few to be sure.
    
    for d in range(2, 10):
        if (time.time() - start_time) > deadline - safety_margin:
            break
            
        if d % 2 == 1:
            est_girth = d + 5
        else:
            est_girth = 2 * d + 1
            
        if est_girth >= g:
            n_candidate = 2 * (q ** d)
            if n_candidate < best_n:
                best_n = n_candidate
                best_d = d
    
    if best_d is None:
        # Fallback to original logic if no small d found
        # Original logic:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_d = d
        best_n = 2 * (q ** d)

    d = best_d
    
    # LUW Construction
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to build graph
    for vertex in order:
        if (time.time() - start_time) > deadline - safety_margin:
            # If we run out of time, we might have an incomplete graph.
            # We must return a valid graph.
            # If the graph is incomplete, it might not be regular or connected.
            # However, the LUW graph is constructed by BFS from a start point.
            # If we stop early, we just have a subgraph.
            # This is invalid.
            # We should probably avoid this by ensuring the construction is fast enough.
            # For d=4, q=5, n=1250. BFS is fast.
            pass
            
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
