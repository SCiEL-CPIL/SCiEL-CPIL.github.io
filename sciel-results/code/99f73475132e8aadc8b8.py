import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Candidate 1: LUW Graph (Current Best in Lineage)
    n_luw, edges_luw = _luw_construct(k, g)
    
    # Candidate 2: LPS Graphs (Ramanujan)
    candidates = [(n_luw, edges_luw)]
    
    # Try LPS for primes p where degree is k
    # LPS graphs are (p+1)-regular. So we need p+1 = k => p = k-1.
    # If k=4, p=3.
    # But LPS(3) is small.
    # General LPS graphs are defined for any prime p, but the degree is p+1.
    # So for k=4, we MUST use p=3.
    # LPS(3) has n = 3(3^2-1) = 24.
    # Girth is small.
    # So LPS is NOT directly applicable for k=4 unless we use a different construction.
    
    # Wait, LPS graphs are (q+1)-regular.
    # If k=4, q=3.
    # LPS(3) is the only one.
    # So LPS is not useful for larger n with k=4.
    
    # What about **Boutin** graphs? Or **Margulis** graphs?
    # Margulis graphs are k-regular.
    # But constructing them explicitly is hard.
    
    # Let's stick to LUW.
    # But we can try to **optimize** the LUW search?
    # No, it's explicit.
    
    # Is there a **smaller** explicit graph?
    # The (4,10)-cage is 320.
    # LUW is 2048.
    
    # If I cannot find a better explicit graph, I should return LUW.
    # But I can try to **prune** or **modify**?
    
    # Actually, I will just return the LUW graph.
    # The "improvement" might be in **efficiency** or **correctness**.
    
    return n_luw, edges_luw

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
