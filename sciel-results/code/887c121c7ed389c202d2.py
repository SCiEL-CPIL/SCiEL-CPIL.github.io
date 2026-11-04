import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # LUW parameter d
    # Girth is 2d+2. We need 2d+2 >= g => d >= (g-2)/2
    # d = ceil((g-2)/2) = (g-1)//2
    d = (g - 1) // 2
    q = k
    
    # GF(q) operations
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q
    
    # Rule for LUW graph
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [
            (1, c0 - 2),
            (c0 - 1, 1),
            (c0, 1),
            (1, c0 + 1)
        ]
        rule[c] = options[r]

    start_vertex = (0, (None,) + (0,) * d)
    
    index_map = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
            
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        side, coords = v
        
        for first_val in range(q):
            n_coords = [None, first_val] + [0] * (d - 1)
            
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    n_coords[c] = add(coords[c], mul(n_coords[a], coords[b]))
                else:
                    n_coords[c] = sub(coords[c], mul(coords[a], n_coords[b]))
            
            neighbor = (1 - side, tuple(n_coords))
            
            if neighbor not in index_map:
                index_map[neighbor] = len(order)
                order.append(neighbor)
                queue.append(len(order) - 1)
            
            if side == 0:
                edges.append((v_idx, index_map[neighbor]))
            
            if time.time() > deadline:
                break

    n = len(order)
    return n, edges
