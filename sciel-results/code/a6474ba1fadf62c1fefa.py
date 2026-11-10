import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using a Cayley graph search.
    For k=3, we use a Cayley graph of PSL(2, p) with generators {A, B, B^-1}.
    """
    if k != 3:
        # Fallback to LUW for other k
        return _luw_construct(k, g)

    # Try different primes p to find a graph with girth >= g
    # Start with smaller p to minimize N
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    
    for p in primes:
        # Construct PSL(2, p)
        # Elements are matrices [[a, b], [c, d]] with ad-bc=1, mod p, modulo -I
        # We represent elements as tuples (a, b, c, d)
        
        def mat_mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return (
                (a1*a2 + b1*c2) % p,
                (a1*b2 + b1*d2) % p,
                (c1*a2 + d1*c2) % p,
                (c1*b2 + d1*d2) % p
            )

        def mat_inv(m):
            a, b, c, d = m
            # Inverse of [[a, b], [c, d]] is [[d, -b], [-c, a]]
            return (d, (-b) % p, (-c) % p, a)

        def mat_eq(m1, m2):
            # Check if m1 == m2 or m1 == -m2 (since PSL is mod -I)
            return (m1 == m2) or (m1 == mat_mul((p-1, 0, 0, p-1), m2)) # -I is (p-1, 0, 0, p-1)

        # Generate all elements of PSL(2, p)
        # This is O(p^4) which is too slow for large p.
        # For p=11, 11^4 = 14641. OK.
        # For p=13, 13^4 = 28561. OK.
        
        elements = []
        seen = set()
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    for d in range(p):
                        if (a*d - b*c) % p == 1:
                            m = (a, b, c, d)
                            # Normalize: ensure a unique representative for +/- I
                            # We can just use the matrix as is, but identify m and -m
                            # To avoid duplicates in the list, we can skip if -m is already seen?
                            # Or just add all, and divide by 2 later?
                            # Simpler: Add all SL(2,p) elements. The Cayley graph of SL(2,p) is a cover of PSL(2,p).
                            # If we use SL(2,p), the size is 2*|PSL|.
                            # Let's use SL(2,p) to avoid quotient logic.
                            elements.append(m)
        
        # Wait, SL(2,p) has order p(p^2-1).
        # For p=11, size 1320.
        # For p=7, size 336.
        
        # Map elements to indices
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        
        # Find involutions (order 2) in SL(2,p)
        # m^2 = I
        involutions = []
        for m in elements:
            m2 = mat_mul(m, m)
            if m2 == (1, 0, 0, 1):
                involutions.append(m)
        
        # Find elements of order > 2
        # We need B such that B != B^-1
        candidates_B = []
        for m in elements:
            if m == (1, 0, 0, 1) or m == (p-1, 0, 0, p-1):
                continue
            if m in involutions:
                continue
            candidates_B.append(m)
        
        if not involutions or not candidates_B:
            continue
            
        # Random search for A, B
        found = False
        # Limit iterations to avoid timeout
        for _ in range(1000):
            A = random.choice(involutions)
            B = random.choice(candidates_B)
            B_inv = mat_inv(B)
            
            # Check if A, B, B_inv are distinct (as generators)
            # A != B, A != B_inv, B != B_inv
            if A == B or A == B_inv:
                continue
            
            # Build graph
            n = len(elements)
            edges = []
            # BFS to check girth
            # Neighbors of i are elem_to_idx[mat_mul(elements[i], A)], ...
            
            # Precompute neighbors for speed?
            # Or just BFS on the fly.
            
            # Check girth
            # BFS from each vertex? Too slow.
            # BFS from one vertex gives shortest cycle containing that vertex.
            # We need global girth.
            # Approximation: Check BFS from a few random vertices.
            # Or: BFS from all vertices is O(N^2). N=336 -> 110k. N=1320 -> 1.7M.
            # 1.7M ops is fine in Python if optimized.
            
            # Let's do BFS from all vertices to be sure.
            # But we can early exit if we find a cycle < g.
            
            ok = True
            # To speed up, we can just check if any cycle < g exists.
            # We can run BFS from every vertex, but stop if we find a back edge with depth < g.
            
            # Optimization: If N is small, just do it.
            # If N is large, sample.
            
            # Let's try BFS from all vertices.
            # For each start s:
            #   dist[s] = 0
            #   parent[s] = -1
            #   Queue
            #   For each neighbor v of u:
            #      if v == parent[u]: continue
            #      if dist[v] == -1:
            #         dist[v] = dist[u] + 1
            #         parent[v] = u
            #      else:
            #         cycle_len = dist[u] + dist[v] + 1
            #         if cycle_len < g:
            #             ok = False
            #             break
            
            # This is O(N * E) = O(N^2) for regular graphs.
            
            # We need to map element to index quickly.
            # neighbors of i:
            # n1 = elem_to_idx[mat_mul(elements[i], A)]
            # n2 = elem_to_idx[mat_mul(elements[i], B)]
            # n3 = elem_to_idx[mat_mul(elements[i], B_inv)]
            
            # Precompute neighbor lists for this (A, B) pair?
            # Yes, takes O(N).
            
            neigh = [[] for _ in range(n)]
            for i in range(n):
                e = elements[i]
                n1 = elem_to_idx[mat_mul(e, A)]
                n2 = elem_to_idx[mat_mul(e, B)]
                n3 = elem_to_idx[mat_mul(e, B_inv)]
                neigh[i] = [n1, n2, n3]
            
            # Check girth
            # We can use a bitset or array for distances.
            # Since we restart BFS for each vertex, we need to reset dist.
            # N=1320. 1320 * 1320 = 1.7M.
            
            for s in range(n):
                dist = [-1] * n
                dist[s] = 0
                parent = [-1] * n
                q = deque([s])
                while q:
                    u = q.popleft()
                    for v in neigh[u]:
                        if v == parent[u]:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            # Found a cycle
                            # The cycle length is dist[u] + dist[v] + 1
                            # But this is only valid if the path from s to u and s to v are disjoint?
                            # In BFS, the first time we see a visited node, the cycle length is dist[u] + dist[v] + 1.
                            # This is the shortest cycle passing through s?
                            # Actually, BFS from s finds the shortest cycle containing s.
                            # The global girth is the minimum over all s.
                            
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Construct edge list
                edge_set = set()
                for i in range(n):
                    for j in neigh[i]:
                        if i < j:
                            edge_set.add((i, j))
                edges = list(edge_set)
                return n, edges
        
    # If no graph found, fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original LUW code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = None
            y[1] = first
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
    from collections import deque
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
