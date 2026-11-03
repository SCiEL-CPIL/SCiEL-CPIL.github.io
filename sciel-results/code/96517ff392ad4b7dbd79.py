import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 4 and g == 10:
        return _construct_4_10(start_time, deadline)
    
    return _luw_construct(k, g)

def _construct_4_10(start_time, deadline):
    # Base graph: K_{5,5} minus perfect matching
    # Top: 0-4, Bottom: 5-9
    # Edges: (i, 5+j) for i != j
    
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
    
    n_lift = 320
    group_size = 32
    
    # Strategy 1: Try known structured voltage assignments
    # The (4,10)-cage is a Z_32-lift of K_{5,5} \ M.
    # One known voltage assignment uses powers of 2.
    
    # Try pattern: v[i][j] = 2^((j-i) % 5) if (j-i) % 5 != 0
    # Mapping: diff 1->2, 2->4, 3->8, 4->16
    patterns = []
    
    # Pattern A: 2, 4, 8, 16
    p = [[0]*5 for _ in range(5)]
    for i in range(5):
        for j in range(5):
            if i != j:
                diff = (j - i) % 5
                if diff == 1: p[i][j] = 2
                elif diff == 2: p[i][j] = 4
                elif diff == 3: p[i][j] = 8
                elif diff == 4: p[i][j] = 16
    patterns.append(p)
    
    # Pattern B: 1, 2, 4, 8 (shifted)
    p = [[0]*5 for _ in range(5)]
    for i in range(5):
        for j in range(5):
            if i != j:
                diff = (j - i) % 5
                if diff == 1: p[i][j] = 1
                elif diff == 2: p[i][j] = 2
                elif diff == 3: p[i][j] = 4
                elif diff == 4: p[i][j] = 8
    patterns.append(p)
    
    # Pattern C: Linear a*i + b*j
    # Try a few small a, b
    for a in [1, 2, 3, 4]:
        for b in [1, 2, 3, 4]:
            p = [[0]*5 for _ in range(5)]
            for i in range(5):
                for j in range(5):
                    if i != j:
                        p[i][j] = (a * i + b * j) % 32
            patterns.append(p)
            
    # Try these patterns
    for p in patterns:
        if time.time() > deadline - 10:
            break
        edges = _build_lift(base_edges, p, group_size)
        if _check_girth_fast(n_lift, edges, 10):
            return n_lift, edges
            
    # Strategy 2: Random search with voltage fixing
    # Fix v[0][1] = 1 to break symmetry
    # Randomize others
    max_attempts = 2000
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        v = [[0]*5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if i != j:
                    if i == 0 and j == 1:
                        v[i][j] = 1
                    else:
                        v[i][j] = random.randint(0, 31)
        
        edges = _build_lift(base_edges, v, group_size)
        if _check_girth_fast(n_lift, edges, 10):
            return n_lift, edges
            
    # Fallback
    return _luw_construct(4, 10)

def _build_lift(base_edges, voltages, group_size):
    edges = []
    for u, v_base in base_edges:
        i = u
        j = v_base - 5
        w = voltages[i][j]
        for x in range(group_size):
            u_id = u * group_size + x
            v_id = v_base * group_size + (x + w) % group_size
            edges.append((u_id, v_id))
    return edges

def _check_girth_fast(n, edges, min_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = min_girth // 2 - 1
    
    for start in range(n):
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
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g):
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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
