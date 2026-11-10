import numpy as np
import time

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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

def _generate_random_regular_graph(n, k):
    """Generate a random simple k-regular graph on n vertices."""
    if n * k % 2 != 0:
        return None
    
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    np.random.shuffle(stubs)
    
    A = np.zeros((n, n), dtype=np.int8)
    valid = True
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            valid = False
            break
        if A[u, v] == 1:
            valid = False
            break
        A[u, v] = 1
        A[v, u] = 1
        
    if not valid:
        return None
        
    return A

def _has_girth_at_least(A, g):
    """Check if graph A has girth at least g."""
    A_pow = A.astype(np.int32).copy()
    for length in range(3, g):
        A_pow = A_pow @ A
        if np.any(np.diag(A_pow) > 0):
            return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Calculate Moore bound
    moore = k * sum([(k - 1) ** i for i in range((g - 3) // 2 + 1)])
    
    # Start search from a reasonable lower bound
    # For k=5, g=9, moore=425.
    # Random graphs likely need n > (k-1)^(g/2) = 4^4.5 ~ 1024
    # But let's try from moore up to 1249.
    # To save time, we can skip if n is too small.
    # However, the check is fast.
    
    # Ensure n is even for k=5
    start_n = max(moore, 100)
    if start_n % 2 != 0:
        start_n += 1
        
    # Upper bound for search: try to beat LUW (1250)
    # We can go up to 1249.
    # But if we find a valid graph, we return it immediately.
    # We want the SMALLEST n.
    # So we iterate n from start_n upwards.
    
    for n in range(start_n, 1250, 2):
        if time.time() > deadline:
            break
            
        # Try a few random graphs
        # If n is close to the threshold, more tries might be needed.
        # But let's keep it simple: 5 tries per n.
        for _ in range(5):
            if time.time() > deadline:
                break
                
            A = _generate_random_regular_graph(n, k)
            if A is None:
                continue
                
            if _has_girth_at_least(A, g):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if A[i, j] == 1:
                            edges.append((i, j))
                return n, edges
                
    # Fallback to LUW construction
    return _luw_construct(k, g)
