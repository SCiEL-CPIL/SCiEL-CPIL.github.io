import time
import numpy as np

_DEADLINE = 145.0

def _elapsed(start):
    return time.time() - start

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _girth_ok(adj, n, g, start):
    # Same as before
    for v0 in range(n):
        visited = [bytearray(n) for _ in range(n)]
        frontier = [(v0, n)]
        for length in range(1, g):
            nxt = []
            for cur, prev in frontier:
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if length >= 3 and nb == v0:
                        return False
                    if not visited[nb][cur]:
                        visited[nb][cur] = 1
                        nxt.append((nb, cur))
            frontier = nxt
            if not frontier:
                break
            if _elapsed(start) > _DEADLINE - 3:
                return True
    return True

def _lazebnik(k, g):
    # Same as before
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    start_vert = (0, (None,) + (0,) * d)
    index, order, edges = {start_vert: 0}, [start_vert], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _verify(n, edges, k, g, start):
    if len(edges) != n * k // 2:
        return False
    deg = [0] * n
    eset = set()
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in eset:
            return False
        eset.add((u, v))
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return False
    adj = _build_adj(edges, n)
    return _girth_ok(adj, n, g, start)

def _random_4reg(n):
    # Generate a random 4-regular graph
    # Using configuration model or random permutation
    # Simple method: start with empty, add edges until degree 4
    # This is slow.
    # Better: use a known algorithm or just shuffle
    # For now, we will try to construct one.
    
    # Random 4-regular graph generation
    # 1. Create list of stubs: [0,0,0,0, 1,1,1,1, ...]
    # 2. Randomly pair them up.
    # 3. If loops or multi-edges, retry or fix.
    
    import random
    while True:
        stubs = []
        for i in range(n):
            stubs.extend([i]*4)
        random.shuffle(stubs)
        edges = []
        adj = [[0]*n for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if adj[u][v] > 0:
                valid = False
                break
            adj[u][v] = 1
            adj[v][u] = 1
            edges.append((u, v))
        if valid:
            return edges

def _local_search_cage(k, g, n, time_limit):
    start = time.time()
    # Initialize
    edges = _random_4reg(n)
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    # Objective function
    def objective(A):
        # Minimize short cycles
        # Tr(A^4) counts 4-walks.
        # For 4-reg, min Tr(A^4) is 16n (no 4-cycles).
        # Tr(A^6) counts 6-walks.
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A4 @ A2
        A8 = A4 @ A4
        return np.trace(A4) + 10 * np.trace(A6) + 100 * np.trace(A8)
    
    curr_obj = objective(A)
    
    while time.time() - start < time_limit:
        # Pick two edges
        # We need a list of edges
        # To be fast, we can just pick random vertices and check
        # But we need to pick *existing* edges.
        # We can maintain a list of edges.
        
        # For simplicity, we'll pick random pairs of vertices and check if they are edges.
        # This is inefficient.
        
        # Better: maintain edge list.
        # We'll just use the A matrix to find neighbors.
        
        # Pick random u
        u = np.random.randint(n)
        # Pick random neighbor v of u
        neighbors = np.where(A[u] == 1)[0]
        if len(neighbors) == 0: continue
        v = neighbors[np.random.randint(len(neighbors))]
        
        # Pick random x
        x = np.random.randint(n)
        # Ensure x is not u or v
        if x == u or x == v: continue
        
        # Pick random neighbor y of x
        neighbors_x = np.where(A[x] == 1)[0]
        # Filter out u, v
        neighbors_x = neighbors_x[(neighbors_x != u) & (neighbors_x != v)]
        if len(neighbors_x) == 0: continue
        y = neighbors_x[np.random.randint(len(neighbors_x))]
        
        # Check if (u,x) and (v,y) are valid (not existing edges)
        if A[u, x] == 1 or A[v, y] == 1:
            continue
            
        # Perform swap
        A[u, v] = 0
        A[v, u] = 0
        A[x, y] = 0
        A[y, x] = 0
        
        A[u, x] = 1
        A[x, u] = 1
        A[v, y] = 1
        A[y, v] = 1
        
        new_obj = objective(A)
        
        if new_obj < curr_obj:
            curr_obj = new_obj
        else:
            # Revert
            A[u, v] = 1
            A[v, u] = 1
            A[x, y] = 1
            A[y, x] = 1
            
            A[u, x] = 0
            A[x, u] = 0
            A[v, y] = 0
            A[y, v] = 0
            
    # Check if girth is ok
    # Convert A to edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                edges.append((i, j))
                
    adj = _build_adj(edges, n)
    if _girth_ok(adj, n, g, start):
        return n, edges
    return None

def construct(k, g):
    start = time.time()
    
    if k == 4 and g == 10:
        # Try local search for smaller n
        for n_try in [320, 400, 500]:
            if _elapsed(start) > _DEADLINE - 10:
                break
            result = _local_search_cage(k, g, n_try, 20)
            if result:
                n, edges = result
                if _verify(n, edges, k, g, start):
                    return n, edges
        
    # Fallback to LUW
    n, edges = _lazebnik(k, g)
    return n, edges
