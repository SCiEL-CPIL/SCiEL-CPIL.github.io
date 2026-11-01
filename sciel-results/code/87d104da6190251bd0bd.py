import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    q = k
    
    d_odd = g - 5
    if d_odd < 3: d_odd = 3
    if d_odd % 2 == 0: d_odd += 1
    d_even = g - 4
    if d_even < 4: d_even = 4
    if d_even % 2 == 1: d_even += 1
    d = min(d_odd, d_even)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    if q == 4:
        mul_table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    else:
        mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]
    
    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i - 1] * q
    total_per_side = q_powers[d]
    total_vertices = 2 * total_per_side
    
    def encode(side, x1_to_d):
        code = side * total_per_side
        for c in range(1, d + 1):
            code += x1_to_d[c - 1] * q_powers[d - c]
        return code
    
    # Precompute all neighbors
    # For side 0, vertex with coords x1..xd:
    #   For first in 0..q-1:
    #     y[1] = first
    #     for c in 2..d: y[c] = (x[c] + mul(y[a], x[b])) % q
    # For side 1, vertex with coords x1..xd:
    #   For first in 0..q-1:
    #     y[1] = first
    #     for c in 2..d: y[c] = (x[c] - mul(x[a], y[b])) % q
    
    # Build adjacency list using integer codes
    adj = [[] for _ in range(total_vertices)]
    
    # Process side 0 vertices
    for code in range(total_per_side):
        # Decode
        x = [0] * d
        rem = code
        for c in range(d, 0, -1):
            x[c - 1] = rem // q_powers[c - 1]
            rem %= q_powers[c - 1]
        
        for first in range(q):
            y = [0] * d
            y[0] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c - 1] = (x[c - 1] + mul_table[y[a - 1]][x[b - 1]]) % q
            other_code = total_per_side + encode(0, y)
            # Wait, encode already adds side offset. Let me fix.
            other_code = encode(1, y)
            adj[code].append(other_code)
            adj[other_code].append(code)
    
    # Process side 1 vertices
    for code in range(total_per_side, 2 * total_per_side):
        offset = code - total_per_side
        x = [0] * d
        rem = offset
        for c in range(d, 0, -1):
            x[c - 1] = rem // q_powers[c - 1]
            rem %= q_powers[c - 1]
        
        for first in range(q):
            y = [0] * d
            y[0] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c - 1] = (x[c - 1] - mul_table[x[a - 1]][y[b - 1]]) % q
            other_code = encode(0, y)
            adj[code].append(other_code)
            adj[other_code].append(code)
    
    # BFS to get connected component (should be all vertices if connected)
    visited = [False] * total_vertices
    order = []
    stack = [0]
    visited[0] = True
    while stack:
        v = stack.pop()
        order.append(v)
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                stack.append(u)
    
    n = len(order)
    # Remap vertices
    remap = {v: i for i, v in enumerate(order)}
    edges = []
    for v in order:
        for u in adj[v]:
            if remap[v] < remap[u]:
                edges.append((remap[v], remap[u]))
    
    return n, edges
