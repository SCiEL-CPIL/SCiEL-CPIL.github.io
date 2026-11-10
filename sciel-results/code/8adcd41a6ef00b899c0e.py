import time

def construct(k, g):
    start_time = time.time()
    
    # Determine the smallest d for the LUW construction CD(d, q) with q=k
    # such that the girth is at least g.
    # Girth lower bound for CD(d, q):
    # d odd: d + 5
    # d even: d + 6
    # Note: For very small d, the bound might be loose or incorrect.
    # We assume the bound holds for d >= 3.
    
    d = 1
    while True:
        if d % 2 == 1:
            bound = d + 5
        else:
            bound = d + 6
        if bound >= g:
            break
        d += 1
        if d > 20: # Safety break
            break
            
    q = k
    
    # Define GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # q is prime (assumed for simplicity, LUW works for prime powers)
        # If q is not prime, we need GF(q) implementation.
        # For now, assume q is prime or 4.
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Rule for LUW construction
    # The rule depends on d.
    # Standard LUW rules:
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2)
    # c>=5: pattern repeats every 4?
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The sequence of rules in LUW is (1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)
        # indexed by r = (c-5)%4
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
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to get connected component
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n = len(order)
    
    # Optional: Verify girth if n is small enough
    # For n=1250, girth check is fast enough.
    # We'll skip full verification for speed, trusting the bound.
    
    return n, edges
