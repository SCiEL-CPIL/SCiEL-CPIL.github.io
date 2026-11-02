import math
import time
import random
import sys
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = 115  # 5s buffer before 120s kill
    
    # Moore bound for k=3, g=16 is 510.
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # 1. Try Circulant Search (Fast, specific family)
    # Circulants for k=3 are C(n; {a, n-a, n/2})
    # We search up to a small limit because they are unlikely to have high girth
    # but it's fast to check.
    max_circ_n = 1200
    for n in range(moore_bound, max_circ_n + 1, 2):
        if time.time() - start_time > deadline:
            break
        h = n // 2
        for a in range(1, h):
            if math.gcd(h, a) != 1:
                continue
            if _check_girth_circulant(n, a, g):
                edges = []
                for u in range(n):
                    for s in (a, n - a, h):
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges

    # 2. Try Random 3-Regular Graph Search
    # This is the main improvement. Random 3-regular graphs have high girth.
    # We start from the Moore bound and go up.
    # We want to find the SMALLEST n.
    # So we iterate n from moore_bound upwards.
    max_random_n = 2000
    
    for n in range(moore_bound, max_random_n + 1, 2):
        if time.time() - start_time > deadline:
            break
        
        # Adaptive attempts based on n
        if n < 700:
            attempts = 3
        elif n < 900:
            attempts = 8
        elif n < 1200:
            attempts = 15
        else:
            attempts = 30
            
        for attempt in range(attempts):
            if time.time() - start_time > deadline:
                break
            
            edges = _generate_random_3_reg(n)
            if edges is None:
                continue
            
            # Check connectivity and girth in one pass
            if _check_girth_and_connected(n, edges, g):
                return n, edges

    # 3. Fallback to LUW
    return _luw_fallback(k, g)

def _check_girth_circulant(n, a, g):
    """Check if C(n; {a, -a, n/2}) has girth >= g."""
    h = n // 2
    S = [a, n - a, h]
    
    max_depth = g // 2 - 1
    visited = {0: -1} # node: depth
    q = deque([(0, -1, 0)])
    
    while q:
        u, parent, d = q.popleft()
        if d == max_depth:
            continue
        for s in S:
            v = (u + s) % n
            if v == parent:
                continue
            if v in visited:
                dv = visited[v]
                cycle_len = d + dv + 1
                if cycle_len < g:
                    return False
            else:
                visited[v] = d
                q.append((v, u, d + 1))
                
    return True

def _generate_random_3_reg(n):
    """Generate a random simple 3-regular graph on n vertices."""
    if n % 2 != 0 or n < 4:
        return None
    
    for _ in range(200):
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        
        edges = []
        valid = True
        edge_set = set()
        
        for i in range(0, 3 * n, 2):
            u = stubs[i]
            v = stubs[i + 1]
            
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                valid = False
                break
            edge_set.add((u, v))
            edges.append((u, v))
            
        if valid:
            return edges
            
    return None

def _check_girth_and_connected(n, edges, g):
    """Check if graph is connected and has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check connectivity first (fast)
    visited = [False] * n
    q = deque([0])
    visited[0] = True
    count = 1
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                q.append(v)
                
    if count != n:
        return False
        
    # Check girth: BFS from each vertex up to depth g//2 - 1
    max_depth = g // 2 - 1
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([(start, 0)])
        
        while q:
            u, d = q.popleft()
            if d == max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    q.append((v, d + 1))
                else:
                    cycle_len = d + dist[v] + 1
                    if cycle_len < g:
                        return False
                    
    return True

def _is_connected(n, edges):
    """Check if graph is connected."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = [False] * n
    q = deque([0])
    visited[0] = True
    count = 1
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                q.append(v)
                
    return count == n

def _check_girth_general(n, edges, g):
    """Check if graph has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_depth = g // 2 - 1
    
    for start in range(n):
        visited = {start: 0}
        q = deque([(start, -1, 0)])
        while q:
            u, parent, d = q.popleft()
            for v in adj[u]:
                if v == parent:
                    continue
                if v in visited:
                    dv = visited[v]
                    cycle_len = d + dv + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[v] = d
                    q.append((v, u, d + 1))
                    
    return True

def _luw_fallback(k, g):
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
