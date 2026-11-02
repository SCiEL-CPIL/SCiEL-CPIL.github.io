import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    q = k
    d = max(1, (g - 1) // 2)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    elif q > 1:
        # Check if prime? For safety, assume prime for now as per task context.
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    else:
        # Fallback for k=1 or 0?
        # k=1, girth g. Path graph?
        # If k=1, n=g-1?
        # Let's just return a simple path if k < 2.
        if k == 1:
            n = max(2, g)
            edges = [(i, i+1) for i in range(n-1)]
            return n, edges
        return 0, []

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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS
    # We can add a time check inside the loop if n is large, but n=1250 is small.
    for vertex in order:
        if time.time() > deadline:
            # Return whatever we have? 
            # If we don't have a full graph, it's invalid.
            # But for k=5, g=9, it finishes in <1s.
            pass
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
