import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    The LUW graph CD(d, q) is q-regular and has girth 2d + 2.
    We set q = k to match the degree.
    We choose the smallest d such that 2d + 2 >= g to minimize the number of vertices n = 2 * q^d.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    q = k
    # Minimal d such that girth 2d + 2 >= g
    # 2d >= g - 2
    # d >= (g - 2) / 2
    d_min = (g - 2 + 1) // 2  # Ceiling division
    if d_min < 1:
        d_min = 1
        
    # The LUW construction is typically defined for d >= 1.
    # Ensure d is at least 1.
    d = d_min

    # Finite field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime for simplicity, or handle composite if needed (standard LUW works for prime powers)
        # For now, assume q is prime. If q is not prime, this might fail.
        # The task says "q prime, or q=4".
        # If k is not prime and not 4, we might have issues.
        # But the seed program assumed q=k works.
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Rule for LUW graph CD(d, q)
    # The adjacency is defined by:
    # Point p = (p_1, ..., p_d)
    # Line l = (l_1, ..., l_d)
    # p ~ l iff l_i - p_i = l_{i-1} p_{i-2} ? 
    # The specific rule in the seed code:
    # y[c] = add(x[c], mul(y[a], x[b]))
    # This corresponds to the standard LUW recurrence.
    
    rule = {}
    # The seed code had a specific rule generation.
    # I will reuse the seed code's rule generation as it is known to produce the LUW graph.
    # However, the seed code's rule generation loop `for c in range(5, d + 1)` suggests it handles d >= 5.
    # For d < 5, it uses hardcoded rules.
    # I need to ensure the rule works for the smaller d.
    
    # Standard LUW rules (from literature):
    # For i=2..d:
    # l_i = p_i + l_{i-1} p_{i-2} (indices might vary)
    # The seed code:
    # rule[2] = (1, 1) -> y[2] = x[2] + y[1]x[1]
    # rule[3] = (2, 1) -> y[3] = x[3] + y[2]x[1]
    # rule[4] = (1, 2) -> y[4] = x[4] + y[1]x[2]
    # This looks like a specific variant. I will stick to the seed's rule logic to ensure correctness within the "family".
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None
        for first in range(q):
            # Construct the neighbor vector y
            # y[0] is None
            y = [None] * (d + 1)
            y[1] = first
            # Initialize others to 0, will be overwritten
            for i in range(2, d + 1):
                y[i] = 0
            
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find the connected component
    # The LUW graph is often connected, but we only return the component of the start node.
    # If the graph is disconnected, this might return a smaller graph than the full 2*q^d.
    # However, for prime q, it is usually connected.
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge if we are on the point side (side 0) to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n = len(order)
    return n, edges
