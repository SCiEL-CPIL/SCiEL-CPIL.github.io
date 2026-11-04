import time

def construct(k, g):
    start_time = time.time()
    deadline = 120.0
    
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Rule for CD(d, q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    def get_neighbours(vertex, alpha):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            # Twist factor
            # Standard LUW uses alpha=1
            # Twisted LUW uses alpha^i?
            # Let's try multiplying the term by alpha^(c-2)
            for c in range(2, d + 1):
                a, b = rule[c]
                # Original: y[c] = x[c] + y[a] * x[b]
                # Twisted: y[c] = x[c] + alpha^(c-2) * y[a] * x[b]
                # Or maybe just alpha * y[a] * x[b]?
                # I will try alpha^(c-2)
                factor = pow(alpha, c - 2, q)
                term = mul(mul(factor, y[a]), x[b])
                y[c] = add(x[c], term) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
            
    start = (0, (None,) + (0,) * d)
    best_n = float('inf')
    best_edges = []
    
    # Try all twists
    for alpha in range(1, q):
        if time.time() > deadline - 5:
            break
            
        index = {start: 0}
        order = [start]
        edges = []
        
        for i in range(len(order)):
            vertex = order[i]
            for other in get_neighbours(vertex, alpha):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        if len(order) < best_n:
            best_n = len(order)
            best_edges = edges
            
    return best_n, best_edges
