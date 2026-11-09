import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth):
    """Check if girth >= target_girth. Returns False early if a shorter cycle is found."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _bfs_girth(adj, n):
    """Compute exact girth using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 2:
                        return min_girth
    return min_girth

def _construct_cage_3_16_cayley():
    """Search for a 3-regular graph of girth >= 16 using Cayley graphs on Z_n x Z_2."""
    # Moore bound for k=3, g=16 is 765.
    # We search n from 765 upwards.
    # The graph is defined on Z_n x Z_2.
    # Generators: a=(0,1), b=(k,1), b_inv=(-k,1).
    # This creates a 3-regular graph.
    deadline = time.time() + 110.0
    
    for n in range(765, 2000):
        if time.time() > deadline:
            break
        # n must be such that the graph is simple.
        # a=(0,1) is always order 2.
        # b=(k,1). b^2 = (2k, 0).
        # If b^2 = e=(0,0), then 2k=0 mod n and 0=0.
        # If 2k=0 mod n, then b has order 2.
        # Then S={a, b, b}. Not a set of 3 distinct elements?
        # If b has order 2, then b=b_inv.
        # Then S={a, b}. Degree 2?
        # No, neighbors are x+a, x+b, x+b.
        # So we need b != b_inv, i.e., b^2 != e.
        # So 2k != 0 mod n.
        
        for k in range(1, n // 2 + 1):
            if (2 * k) % n == 0:
                continue
            
            # Build adjacency
            # Vertices are (i, j) mapped to i*2 + j
            # Total vertices N = 2n
            N = 2 * n
            adj = [[] for _ in range(N)]
            
            # Precompute neighbors for efficiency?
            # Or just build and check.
            # Building adj list takes O(N).
            # Girth check takes O(N * (N+E)) ~ O(N^2).
            # N ~ 1000. N^2 = 10^6.
            # 2000 n values * 500 k values = 10^6 checks.
            # 10^6 * 10^6 = 10^12. Too slow.
            
            # We need a faster girth check or fewer candidates.
            # Most k will have small girth.
            # We can do a quick check for small cycles first.
            
            # Let's optimize the girth check.
            # Only check girth if we don't find a short cycle quickly.
            
            # Actually, we can just build the graph and use the existing _has_girth_at_least.
            # But 10^6 graph constructions is slow.
            
            # Let's reduce the search space.
            # Try only specific k values?
            # Or limit n range.
            
            # For now, let's try n up to 1000 and k up to n/2.
            # And add a time check inside the k loop.
            
            # Build adj
            for i in range(n):
                v0 = i * 2
                v1 = i * 2 + 1
                
                # Neighbors of (i,0): (i,1), (i+k,1), (i-k,1)
                u1 = i * 2 + 1
                u2 = ((i + k) % n) * 2 + 1
                u3 = ((i - k) % n) * 2 + 1
                
                adj[v0].append(u1)
                adj[v0].append(u2)
                adj[v0].append(u3)
                
                # Neighbors of (i,1): (i,0), (i+k,0), (i-k,0)
                w0 = i * 2
                w1 = ((i + k) % n) * 2
                w2 = ((i - k) % n) * 2
                
                adj[v1].append(w0)
                adj[v1].append(w1)
                adj[v1].append(w2)
            
            if _has_girth_at_least(adj, N, 16):
                edges = []
                for i in range(n):
                    v0 = i * 2
                    v1 = i * 2 + 1
                    edges.append((v0, v1))
                    edges.append((v0, ((i + k) % n) * 2 + 1))
                    edges.append((v0, ((i - k) % n) * 2 + 1))
                return N, edges
                
    return None

def _construct_cage_3_16():
    # ... (keep existing random search, though it's slow)
    n = 936
    half = n // 2
    deadline = time.time() + 110.0
    for attempt in range(5000):
        if time.time() > deadline:
            break
        stubs = []
        for j in range(half):
            stubs.extend([j, j, j])
        random.shuffle(stubs)
        left_adj = [set() for _ in range(half)]
        valid = True
        for i in range(half):
            for offset in range(3):
                j = stubs[i * 3 + offset]
                left_adj[i].add(j)
            if len(left_adj[i]) != 3:
                valid = False
                break
        if not valid:
            continue
        right_degree = [0] * half
        for i in range(half):
            for j in left_adj[i]:
                right_degree[j] += 1
        if any(d != 3 for d in right_degree):
            continue
        adj = [[] for _ in range(n)]
        for i in range(half):
            for j in left_adj[i]:
                adj[i].append(half + j)
                adj[half + j].append(i)
        # Use the faster check
        if _has_girth_at_least(adj, n, 16):
            edges = []
            for i in range(half):
                for j in left_adj[i]:
                    edges.append((i, half + j))
            return n, edges
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction.
    Optimized to find minimal d for required girth."""
    q = k
    # Start with a small d and increase until girth is satisfied
    # Lower bound for d: girth ~ 2d, so d ~ g/2
    # But algebraic constraints might require larger d.
    # Let's try d from 5 to 15.
    
    def _build_luw(d):
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
        
        rule_list = [rule[c] for c in range(2, d + 1)]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for idx, c in enumerate(range(2, d + 1)):
                    a, b = rule_list[idx]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        i = 0
        while i < len(order):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges

    # Try increasing d
    # We need to be careful with time.
    deadline = time.time() + 100.0
    
    for d in range(5, 20):
        if time.time() > deadline:
            break
        
        n, edges = _build_luw(d)
        
        # Build adjacency for girth check
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _has_girth_at_least(adj, n, g):
            return n, edges
        
    # Fallback to original large d
    d_orig = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _build_luw(d_orig)

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0
    
    if k == 3 and g == 16:
        # Try the new Cayley graph search first
        result = _construct_cage_3_16_cayley()
        if result is not None:
            return result
        
        result = _construct_cage_3_16()
        if result is not None:
            return result
            
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
