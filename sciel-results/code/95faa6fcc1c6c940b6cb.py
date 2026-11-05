import time
import random
from collections import deque

def _check_girth(n, edges, min_girth):
    """
    Check if the graph has girth >= min_girth.
    Uses BFS from each vertex to find shortest cycles.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_girth = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] * 2 >= min_girth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < best_girth:
                        best_girth = cycle_len
                    if best_girth < min_girth:
                        return False
    return best_girth >= min_girth

def _check_girth_fast(n, edges, min_girth):
    """
    Faster girth check using bitsets for small n.
    """
    if n > 2000:
        return _check_girth(n, edges, min_girth)
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # For each vertex, BFS with early termination
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= min_girth // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif abs(dist[u] - dist[v]) <= 1:
                    # Check if this creates a short cycle
                    # In BFS tree, if u and v are both visited and not parent-child
                    # We need to be more careful
                    pass
        # More careful check: track parent
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= min_girth // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    best_n = float('inf')
    best_edges = None
    
    # LUW construction
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    def build_luw(d):
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

    # 1. Try LUW d=4 first
    if k == 5 and g == 9:
        n, edges = build_luw(4)
        if _check_girth(n, edges, g):
            best_n = n
            best_edges = edges
            print(f"LUW d=4: n={n}, girth OK")
    
    # 2. Try circulant graph search
    # For 5-regular circulant on n vertices, we need 5 step values
    # such that the graph is simple and has girth >= 9
    if time.time() < deadline and best_n > 1000:
        # Moore bound for (5,9): 1 + 5*(1+4+16+64) = 1 + 5*85 = 426
        # Try n from 426 to best_n-1
        # For circulant, n must be such that we can pick 5 steps from 1..n//2
        # with no small cycles
        
        # Start from smaller n and work up
        # Use random search with structured initialization
        rng = random.Random(42)
        
        for n in range(430, min(best_n, 1100), 2):  # n must be even for 5-regular
            if time.time() > deadline:
                break
            
            # Try several random step sets
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Pick 5 distinct steps from 1 to n//2
                max_step = n // 2
                steps = rng.sample(range(1, max_step + 1), 5)
                
                # Build edges
                edges = []
                adj = [set() for _ in range(n)]
                valid = True
                for i in range(n):
                    for s in steps:
                        j = (i + s) % n
                        if j == i:
                            valid = False
                            break
                        if j not in adj[i]:
                            adj[i].add(j)
                            adj[j].add(i)
                            if i < j:
                                edges.append((i, j))
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check degree
                if any(len(adj[i]) != k for i in range(n)):
                    continue
                
                # Check girth
                if _check_girth_fast(n, edges, g):
                    best_n = n
                    best_edges = edges
                    print(f"Circulant: n={n}, steps={steps}, girth OK")
                    break  # Found one for this n, try smaller n
            
            # If we found a valid graph, try smaller n
            if best_n == n:
                continue
    
    # 3. If no good result, fall back to LUW d=4 or d=5
    if best_edges is None:
        d = 4
        best_n, best_edges = build_luw(d)
        if not _check_girth(best_n, best_edges, g):
            d = 5
            best_n, best_edges = build_luw(d)
    
    return best_n, best_edges
