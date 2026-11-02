import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Use LUW construction with optimal d
    n, edges = _construct_luw_optimized(k, g)
    
    return n, edges

def _construct_luw_optimized(k, g):
    # Lazebnik-Ustimenko-Woldar construction
    # Degree k = q
    # Girth >= 2d + 1
    # We need 2d + 1 >= g  =>  d >= (g-1)/2
    q = k
    
    # Calculate minimal d
    d = (g - 1) // 2
    if 2 * d + 1 < g:
        d += 1
        
    # Ensure d is at least 3 for the recurrence to work well?
    # LUW usually requires d >= 3.
    if d < 3:
        d = 3

    # Define operations for field GF(q)
    # Since q=k and k is small (e.g., 5), we can use integers mod q
    # Note: LUW works for prime powers. If k is not prime power, this might fail?
    # The task says "small 5-regular graph", implying k=5 is the main case.
    # For general k, if k is not prime power, LUW doesn't directly apply.
    # But the prompt implies we handle the specific case or general k.
    # Assuming k is a prime power for this construction.
    
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q
    
    # Define the rule for the recurrence
    # rule[c] = (a, b) such that y[c] = op(y[a], x[b])
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    val = mul(y[a], x[b])
                    y[c] = add(x[c], val)
                else:
                    val = mul(x[a], y[b])
                    y[c] = sub(x[c], val)
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1)) # y[0] is unused, set to 0
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to generate all vertices
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge if not already added
            # Only add edges from side 0 to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
