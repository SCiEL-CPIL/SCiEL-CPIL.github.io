import time
import random
from collections import defaultdict

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking walks from each vertex."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        # state: (current, prev)
        visited = set()
        stack = [(start, -1, 0)]
        while stack:
            node, prev, dist = stack.pop()
            if dist >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and dist + 1 >= g:
                    continue
                if dist + 1 < g and nb == start:
                    return False
                state = (nb, node)
                if state in visited:
                    continue
                visited.add(state)
                stack.append((nb, node, dist + 1))
    return True

def _check_circulant_girth(n, S, g):
    """Check if circulant graph C_n(S) has girth >= g."""
    # S is a set of generators
    # Build adj
    adj = [[] for _ in range(n)]
    for i in range(n):
        for s in S:
            u = (i + s) % n
            v = (i - s) % n
            # Add edges
            if u != i:
                adj[i].append(u)
            if v != i:
                adj[i].append(v)
            # Remove duplicates and self loops
            # Since S is symmetric and 0 not in S, u!=i, v!=i usually.
            # But if s = n/2, u=v=i+s.
    
    # Deduplicate
    for i in range(n):
        adj[i] = list(set(adj[i]))
        if len(adj[i]) != len(S) * 2: # Expected degree
             # If s=n/2, it contributes 1 edge, not 2?
             # If S = {n/2, a, b}, degree is 1 + 2 + 2 = 5.
             # If s != n/2, degree is 2.
             # So we just check degree == k?
             pass
             
    # Actually, for girth check, we just need the graph.
    # Let's just check girth.
    for start in range(n):
        visited = set()
        stack = [(start, -1, 0)]
        while stack:
            node, prev, dist = stack.pop()
            if dist >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start:
                    if dist + 1 < g:
                        return False
                    else:
                        continue
                state = (nb, node)
                if state in visited:
                    continue
                visited.add(state)
                stack.append((nb, node, dist + 1))
    return True

def _luw_construct(k, g, deadline):
    q = k
    # Smallest d with 2d+2 >= g
    d_min = (g - 2 + 1) // 2
    if d_min < 2: d_min = 2
    # LUW girth is 2d+2.
    # We need 2d+2 >= g.
    # So d >= (g-2)/2.
    # d_min is ceil((g-2)/2).
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d_min + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d_min - 1)
            for c in range(2, d_min + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d_min)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline - 10:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    n = len(order)
    return n, edges

def construct(k, g):
    deadline = time.time() + 120
    best_n = float('inf')
    best_edges = None
    
    # Try Circulants
    # k=5, so S has 5 elements.
    # S = {n/2, a, b, -a, -b}? No, S is the set of jumps.
    # If we define graph by jumps S, degree is |S| if we consider directed?
    # Undirected: edges (i, i+s) for s in S.
    # If s in S, then -s in S for symmetry?
    # If S = {s1, s2, s3, s4, s5}, and graph is undirected,
    # we usually take S symmetric.
    # If k=5 is odd, one jump must be self-inverse (n/2).
    # So S = {n/2, a, b, -a, -b}.
    # This gives degree 5.
    
    # Search n from Moore bound
    moore_bound = 1 + k + k*(k-1) + k*(k-1)**2 + k*(k-1)**3
    # For g=9, radius 4.
    # 1 + 5 + 20 + 80 + 320 = 426.
    
    start_n = 428
    if k == 5 and g == 9:
        # Try a few random n and S
        random.seed(42)
        for n in range(start_n, start_n + 2000, 2):
            if time.time() > deadline - 30:
                break
            # Try a few random pairs a, b
            for _ in range(100):
                if time.time() > deadline - 30:
                    break
                a = random.randint(1, n//2 - 1)
                b = random.randint(1, n//2 - 1)
                if a == b: continue
                S = [n//2, a, b, -a, -b]
                # Check degree
                # n/2 is self-inverse.
                # a, -a are distinct if a != n/2.
                # b, -b are distinct if b != n/2.
                # a != b, a != -b.
                # So 5 distinct jumps.
                # Check girth
                if _check_circulant_girth(n, S, g):
                    # Build edges
                    edges = []
                    for i in range(n):
                        for s in S:
                            u = (i + s) % n
                            if i < u:
                                edges.append((i, u))
                    # Verify
                    if _verify_girth(n, edges, g):
                        return n, edges
                    else:
                        # Should not happen if _check passed
                        pass
                        
    # Fallback to LUW with d=4
    n_luw, edges_luw = _luw_construct(k, g, deadline)
    return n_luw, edges_luw

