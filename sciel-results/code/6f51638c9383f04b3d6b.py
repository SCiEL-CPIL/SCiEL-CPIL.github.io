import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # For k=4, g=10, try the 2-vertex voltage graph construction
    if k == 4 and g == 10:
        result = _voltage_lift_search(deadline)
        if result is not None:
            return result
            
    # Fallback to LUW if voltage search fails or times out
    result = _luw_construct(k, g)
    if result is not None:
        n, edges = result
        if _verify_girth(n, edges, g):
            return n, edges
            
    return _luw_construct(k, g)

def _voltage_lift_search(deadline):
    """
    Search for smallest M and voltages v1..v4 in Z_M such that:
    1. v1, v2, v3, v4 are distinct.
    2. No non-backtracking closed walk of length < 10 has sum 0.
    """
    # Try M from 4 upwards
    # We need at least 4 distinct values, so M >= 4.
    # Known bound is 320 vertices -> M=160.
    # Let's try primes and composites.
    
    # Optimization: The constraints are symmetric. We can fix v1=0, v2=1, v3=a, v4=b.
    # Then search for M, a, b.
    
    for M in range(4, 300):
        if time.time() > deadline - 10:
            break
            
        # If M is small, brute force a, b
        # a in 2..M-1, b in 2..M-1, a != b
        # We can restrict a < b to reduce search space by symmetry? 
        # The set {0, 1, a, b} is what matters.
        
        found = False
        # To speed up, we can check constraints as we go.
        # Constraints:
        # L=2: v_i - v_j != 0 (satisfied by distinctness)
        # L=4: v_i - v_j + v_k - v_l != 0 for all sequences i,j,k,l with i!=j, j!=k, k!=l.
        # Actually, the walk is i1, i2, i3, i4. i1!=i2, i2!=i3, i3!=i4.
        # Sum: v_i1 - v_i2 + v_i3 - v_i4.
        
        # Let's just iterate a, b
        for a in range(2, M):
            for b in range(a + 1, M):
                if time.time() > deadline - 5:
                    break
                    
                voltages = [0, 1, a, b]
                
                # Check L=4
                ok = True
                for i1 in range(4):
                    for i2 in range(4):
                        if i1 == i2: continue
                        for i3 in range(4):
                            if i2 == i3: continue
                            for i4 in range(4):
                                if i3 == i4: continue
                                s = voltages[i1] - voltages[i2] + voltages[i3] - voltages[i4]
                                if s % M == 0:
                                    ok = False
                                    break
                            if not ok: break
                        if not ok: break
                    if not ok: break
                if not ok: continue
                
                # Check L=6
                # i1, i2, i3, i4, i5, i6
                # s = v_i1 - v_i2 + v_i3 - v_i4 + v_i5 - v_i6
                ok = True
                for i1 in range(4):
                    for i2 in range(4):
                        if i1 == i2: continue
                        for i3 in range(4):
                            if i2 == i3: continue
                            for i4 in range(4):
                                if i3 == i4: continue
                                for i5 in range(4):
                                    if i4 == i5: continue
                                    for i6 in range(4):
                                        if i5 == i6: continue
                                        s = voltages[i1] - voltages[i2] + voltages[i3] - voltages[i4] + voltages[i5] - voltages[i6]
                                        if s % M == 0:
                                            ok = False
                                            break
                                    if not ok: break
                                if not ok: break
                            if not ok: break
                        if not ok: break
                    if not ok: break
                if not ok: continue
                
                # Check L=8
                # s = v1 - v2 + v3 - v4 + v5 - v6 + v7 - v8
                ok = True
                for i1 in range(4):
                    for i2 in range(4):
                        if i1 == i2: continue
                        for i3 in range(4):
                            if i2 == i3: continue
                            for i4 in range(4):
                                if i3 == i4: continue
                                for i5 in range(4):
                                    if i4 == i5: continue
                                    for i6 in range(4):
                                        if i5 == i6: continue
                                        for i7 in range(4):
                                            if i6 == i7: continue
                                            for i8 in range(4):
                                                if i7 == i8: continue
                                                s = (voltages[i1] - voltages[i2] + voltages[i3] - voltages[i4] + 
                                                     voltages[i5] - voltages[i6] + voltages[i7] - voltages[i8])
                                                if s % M == 0:
                                                    ok = False
                                                    break
                                            if not ok: break
                                        if not ok: break
                                    if not ok: break
                                if not ok: break
                            if not ok: break
                        if not ok: break
                    if not ok: break
                if not ok: continue
                
                # Found valid M and voltages
                # Construct the graph
                n = 2 * M
                edges = []
                # Vertices 0..M-1 are A, M..2M-1 are B
                # A_x connected to B_{x + v_i}
                for x in range(M):
                    for v in voltages:
                        y = (x + v) % M
                        u = x
                        w = M + y
                        edges.append((u, w))
                
                # Verify
                if _verify_girth(n, edges, 10):
                    return n, edges
                else:
                    # Should not happen if logic is correct, but safety check
                    pass
                    
        # End for M
        
    return None

def _verify_girth(n, edges, g):
    """Verify graph is 4-regular and has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != 4:
            return False
            
    edge_set = set()
    for u, v in edges:
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        edge_set.add((u, v))
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] >= g:
                        break
                    queue.append(v)
                elif dist[v] != dist[u] + 1:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _luw_construct(k, g):
    """
    LUW construction fallback.
    """
    q = k
    d = (g + 1) // 2
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - r
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
