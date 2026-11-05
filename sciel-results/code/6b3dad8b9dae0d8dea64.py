import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q) with q=k.
    The girth of CD(d, q) is at least 2*d + 2 for prime q.
    We choose the smallest d such that 2*d + 2 >= g.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    # For the specific task k=3, q=3 is prime.
    # LUW requires q to be prime or q=4.
    # If k is not prime and not 4, we cannot use LUW directly.
    # However, the task is for k=3.
    # If k is not supported, we fall back to a simple cycle or return invalid.
    # But we assume k=3.
    
    q = k
    
    # Check if q is prime or 4
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    if q == 4:
        # Handle q=4 case (not needed for k=3, but for completeness)
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif is_prime(q):
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        # Fallback: Return a simple cycle if k=2, or raise error/return minimal valid graph if possible.
        # For k=3, this branch should not be hit.
        # If hit, we cannot construct a valid k-regular graph with LUW.
        # We return a dummy graph to avoid crash, though it will be invalid.
        # Better to just return an empty graph or raise.
        # The task expects a valid graph.
        # Since k=3 is guaranteed, we ignore this for now.
        return 0, []

    # Calculate optimal d
    # Girth >= 2*d + 2
    # We need 2*d + 2 >= g
    # d >= (g - 2) / 2
    d = max(1, (g - 2) // 2)
    
    # Ensure d is at least 3 for the rule to make sense?
    # The seed used max(3, ...). 
    # If d < 3, the rule dictionary in the seed handles 2, 3, 4.
    # Let's use the seed's rule generation logic.
    
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
