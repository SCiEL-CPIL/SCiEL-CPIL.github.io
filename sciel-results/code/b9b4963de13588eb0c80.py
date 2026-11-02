import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # If the task is specifically for the (4, 10)-cage, try to construct it
    if k == 4 and g == 10:
        return _construct_4_10(start_time, deadline)
    
    # Fallback for other parameters
    return _luw_construct(k, g)

def _construct_4_10(start_time, deadline):
    # Base graph: K_{5,5} minus perfect matching
    # Vertices 0..4 (top), 5..9 (bottom)
    # Edges (i, 5+j) for i != j
    
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
    
    n_lift = 320
    group_size = 32
    
    # Strategy 1: Try known structured voltage assignments
    # The (4,10)-cage can be constructed as a Z_32-lift of K_{5,5} minus perfect matching
    # Try various structured patterns
    
    # First, try a deterministic structured approach
    # Use voltages based on a linear function: v[i][j] = (a*i + b*j + c) % 32
    for a in range(1, 32):
        for b in range(1, 32):
            for c in range(32):
                if time.time() > deadline - 10:
                    break
                v = [[0]*5 for _ in range(5)]
                for i in range(5):
                    for j in range(5):
                        if i != j:
                            v[i][j] = (a * i + b * j + c) % 32
                
                edges = []
                for u, v_base in base_edges:
                    i = u
                    j = v_base - 5
                    w = v[i][j]
                    for x in range(32):
                        u_id = u * 32 + x
                        v_id = v_base * 32 + (x + w) % 32
                        edges.append((u_id, v_id))
                
                if _check_girth_fast(n_lift, edges, 10):
                    return n_lift, edges
        if time.time() > deadline - 10:
            break
    
    # Strategy 2: Random search with more attempts
    max_attempts = 5000
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Assign random voltages
        v = [[0]*5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if i != j:
                    v[i][j] = random.randint(0, 31)
        
        # Build lift edges
        edges = []
        for u, v_base in base_edges:
            i = u
            j = v_base - 5
            w = v[i][j]
            for x in range(32):
                u_id = u * 32 + x
                v_id = v_base * 32 + (x + w) % 32
                edges.append((u_id, v_id))
        
        # Check girth
        if _check_girth_fast(n_lift, edges, 10):
            return n_lift, edges
    
    # If all searches fail, fallback to LUW construction
    return _luw_construct(4, 10)

def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = min_girth // 2 - 1
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            d = dist[u]
            if d >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a back edge - check cycle length
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < min_girth:
                        return False
    
    return True

def _check_girth(n, edges, min_girth):
    """Check girth using BFS - kept for compatibility but use _check_girth_fast."""
    return _check_girth_fast(n, edges, min_girth)

def _luw_construct(k, g):
    """LUW construction - produces a graph with girth >= g but possibly many vertices.
    This is used as a fallback when the 320-vertex construction fails."""
    q = k
    # For the LUW construction CD(d,q), girth is at least d+5
    # We need d+5 >= g, so d >= g-5
    # For odd g, d = g-5; for even g, d = g-4 (to ensure proper girth)
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        # GF(4) arithmetic
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rules for the LUW construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

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
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
