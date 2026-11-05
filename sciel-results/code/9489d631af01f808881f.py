import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Strategy 1: Try to find a small Circulant or Abelian Cayley graph
    # We know n must be >= 512.
    # Let's search n from 512 to 2000 for circulant graphs C_n({1, a, b})
    
    best_n = None
    best_edges = None
    
    # Helper to check girth
    def check_girth(n, shifts):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for s in shifts:
                j = (i + s) % n
                if i < j:
                    adj[i].append(j)
                    adj[j].append(i)
                elif i > j:
                    # Avoid duplicates, but for girth check we just need adjacency
                    pass
        
        # Actually, for girth, we can just do BFS from each node?
        # Too slow for n=1000.
        # We can check for short cycles specifically.
        # Girth >= 16 means no cycles of length 4, 6, 8, 10, 12, 14.
        
        # Check 4-cycles
        # x ~ x+s1 ~ x+s1+s2 ~ x+s2
        # (x+s1) + s2 = x + s1 + s2
        # (x+s2) + s1 = x + s1 + s2
        # So a 4-cycle exists if there are s1, s2, s3, s4 in shifts such that s1+s2 = s3+s4
        # and the path is simple.
        
        # This is complex. Let's just use BFS for small n.
        # For n up to 1000, BFS from every node is O(n * (n+E)) ~ 1000 * 3000 = 3M ops.
        # This is fast enough in Python if optimized.
        
        min_girth = float('inf')
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        # We can prune if min_girth <= 4? No, we need >= 16.
                        # If we find a cycle < 16, we can stop early for this start?
                        # No, we need the global min.
                        # But if we find a cycle of length 4, girth is 4.
                        # We can break early if min_girth <= g-1.
                        if min_girth < g:
                            return min_girth
                            
        return min_girth

    # Search Circulants
    # n must be even? No.
    # Shifts must be distinct and non-zero.
    # Also, to be 3-regular, we need 3 shifts.
    # If we use {1, a, b}, we need 1, a, b to be distinct mod n.
    # And no shift is 0.
    
    for n in range(512, 2000):
        if time.time() > deadline:
            break
        # Only check if n is large enough?
        # Moore bound is 512.
        
        # Try a few random shifts
        # Or systematic?
        # Systematic is better for reproducibility.
        # But 1000 * 1000 * 1000 is too big.
        # Let's try random shifts.
        
        for _ in range(100): # 100 trials per n
            if time.time() > deadline:
                break
            shifts = [1]
            a = random.randint(2, n-1)
            b = random.randint(2, n-1)
            if a == b or a == 1 or b == 1:
                continue
            shifts = [1, a, b]
            
            # Ensure symmetry? No, undirected graph edges are (i, i+s) and (i, i-s).
            # So shifts {1, a, b} generates edges for +/-1, +/-a, +/-b.
            # If a = b, duplicate.
            # If a = -b, then edge (i, i+a) is same as (i, i-b)? No, (i, i+a) and (i, i-b) are different unless a=b.
            # Wait, in undirected graph, edge {u, v} is same as {v, u}.
            # So shift s and -s produce the same edges.
            # So we just need to pick 3 distinct "absolute" shifts?
            # No, we pick 3 shifts $s_1, s_2, s_3$.
            # The edges are $(i, i+s_j)$ for all $i, j$.
            # Since the graph is undirected, the set of edges is determined by the set $\{s_1, s_2, s_3\}$ modulo sign.
            # So we should pick $s_1, s_2, s_3$ such that $s_i \not\equiv \pm s_j$.
            
            # Check for 4-cycles quickly?
            # A 4-cycle exists if $s_i + s_j \equiv s_k + s_l$?
            # Path: 0 -> s1 -> s1+s2 -> s2.
            # This requires s1+s2 = s2? No.
            # 0 -> s1 -> s1-s2 -> -s2.
            # This is a cycle of length 4 if s1-s2 = -s2 => s1=0 (no).
            # Or s1-s2 = s3?
            
            # Let's just run BFS.
            girth = check_girth(n, shifts)
            if girth >= g:
                best_n = n
                # Reconstruct edges
                edges = []
                for i in range(n):
                    for s in shifts:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    # If no small circulant found, fallback to LUW (from seed)
    # But LUW is huge.
    # Is there a better fallback?
    # Maybe a larger search range for circulants?
    # Or just return the LUW graph.
    
    # Let's try to generate the LUW graph as in the seed, but maybe it's too slow?
    # The seed runs in < 120s.
    # So we can just use the seed's logic as fallback.
    
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
