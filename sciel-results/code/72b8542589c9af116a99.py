import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k != 4:
        return 0, []
        
    # Try LUW constructions with various q, d
    # Candidates: (q, d)
    # We want small n = 2*q^d, and girth >= g
    candidates = [
        (5, 3), # n=250
        (7, 3), # n=686
        (5, 4), # n=1250
        (11, 3), # n=2662
        (4, 6), # n=2560 (Seed's likely config)
        (13, 3), # n=4394
        (7, 4), # n=4802
    ]
    
    best_n = float('inf')
    best_edges = None
    
    for q, d in candidates:
        if time.time() > deadline - 10:
            break
        n = 2 * (q ** d)
        if n >= best_n:
            continue
            
        edges = _luw_construct_general(q, d)
        if edges is None:
            continue
            
        if _verify(n, edges, k, g):
            if n < best_n:
                best_n = n
                best_edges = edges
                
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to seed's logic if all fail (should not happen)
    return _luw_construct_seed(k, g)

def _luw_construct_general(q, d):
    """
    Constructs LUW graph for prime q and depth d.
    Vertices: (side, x1, ..., xd)
    """
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q
    
    # Rules: rule[c] = (a, b) means y_c = x_c + y_a * x_b
    # Standard LUW rules for k=4
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    # For c >= 5, use cyclic pattern from seed
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # Seed pattern: [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
        # Note: indices must be valid (1..d)
        # c0 is roughly c
        # Let's stick to the seed's pattern for consistency
        pattern = [
            (1, c0 - 2), 
            (c0 - 1, 1), 
            (c0, 1), 
            (1, c0 + 1)
        ]
        # Ensure indices are within 1..d
        # If index out of bounds, this rule might be invalid for small d
        # But d >= 5 here, so c0 is 5,6,7...
        # c0-2 >= 3, c0+1 <= d+1? 
        # If c=d, c0=d. c0+1 = d+1 which is out of bounds for x (1..d).
        # The seed code had `x[b]` where b is from rule.
        # If b > d, it crashes.
        # So this rule is only valid if indices are <= d.
        # For c=d, c0=d.
        # If r=0: (1, d-2) OK.
        # If r=1: (d-1, 1) OK.
        # If r=2: (d, 1) OK.
        # If r=3: (1, d+1) FAIL.
        # So we must handle bounds.
        # For simplicity, only use this if d is large enough or adjust.
        # Given we are testing d=3,4, this loop (c>=5) won't run for d=3,4.
        # For d=6 (seed), c=5,6.
        # c=5: c0=5. r=0. (1,3). OK.
        # c=6: c0=6. r=1. (5,1). OK.
        # So for d=6, it works.
        
        a, b = pattern[r]
        if 1 <= a <= d and 1 <= b <= d:
            rule[c] = (a, b)
        else:
            # Fallback rule if index out of bounds
            rule[c] = (1, 1)
            
    def neighbours(vertex):
        side, x = vertex
        # x is tuple (x1, ..., xd)
        for first in range(q):
            # y is list of length d+1 (index 0 unused)
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c-1], mul(y[a], x[b-1]))
                else:
                    y[c] = sub(x[c-1], mul(x[a-1], y[b]))
            yield 1 - side, tuple(y)
            
    start = (0, tuple([0] * d))
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n = len(order)
    # Verify n matches expected 2*q^d
    if n != 2 * (q ** d):
        return None
        
    edges_list = sorted(set(edges))
    return edges_list

def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1:
        return False
    
    if len(edges) != len(set(edges)):
        return False
        
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # Girth check
    INF = 10 ** 9
    for (u, v) in edges:
        dist = {}
        q = []
        for w in adj[u]:
            if w == v:
                continue
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v:
                if L + 1 < best:
                    best = L + 1
                continue
            if L >= g:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True

def _luw_construct_seed(k, g):
    # Fallback to seed's specific implementation
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
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
                
    n = len(order)
    edges_list = sorted(set(edges))
    return n, edges_list
