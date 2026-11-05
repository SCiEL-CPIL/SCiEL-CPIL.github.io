import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Moore bound for k-regular graph of girth g (even g)
    # n >= 2 * (k * (k-1)^(g/2 - 1) - 1) / (k - 2)
    if k == 2:
        n_min = g
    else:
        r = g // 2
        n_min = int(2 * (k * (k - 1) ** (r - 1) - 1) / (k - 2))
    
    best_n = float('inf')
    best_edges = None
    
    # Try Circulant graphs C(n; 1, -1, m)
    # This is a 3-regular graph if m not in {0, 1, n-1}
    # We search for n >= n_min
    # Limit search range to avoid infinite loop
    max_n_search = n_min + 2000
    
    for n in range(n_min, max_n_search):
        if time.time() > deadline - 10:
            break
            
        # m must be such that edges are distinct
        # Edges: (i, i+1), (i, i-1), (i, i+m)
        # i+m == i+1 => m=1
        # i+m == i-1 => m=n-1
        # i+m == i => m=0
        # Also need simple graph: no multi-edges.
        # (i, i+m) and (j, j+m) are distinct if m != 0.
        # (i, i+1) and (j, j+m) could be same edge?
        # Edge (u, v) is same as (v, u).
        # If m = n/2, edge (i, i+n/2) is same as (i+n/2, i).
        # This is fine, it's just one edge.
        
        for m in range(2, n // 2 + 1):
            if m == 1 or m == n - 1:
                continue
                
            # Check girth
            # BFS from all vertices to depth g//2
            # State: (node, prev_node)
            # If we revisit a state, or if we find a node visited at depth d1 and d2 with d1+d2 < g, cycle exists.
            # Actually, standard BFS from each vertex to depth g//2.
            # If any node is reached at depth < g//2 from two different paths (or same path backtracking), cycle.
            # Simpler: BFS from each vertex, record distance. If a neighbor is already visited and dist[u] + dist[v] + 1 < g, cycle?
            # No, that detects triangles etc.
            # For girth >= g, BFS tree from any vertex to depth g/2 must be a tree (no collisions).
            
            valid = True
            half_g = g // 2
            
            for start in range(n):
                # BFS from start
                # dist[node] = depth
                # prev[node] = previous node in BFS tree
                dist = [-1] * n
                prev = [-1] * n
                dist[start] = 0
                queue = deque([(start, -1)])
                
                while queue:
                    node, p = queue.popleft()
                    d = dist[node]
                    if d >= half_g:
                        continue
                    
                    # Neighbors: node+1, node-1, node+m
                    for nb in [(node + 1) % n, (node - 1) % n, (node + m) % n]:
                        if dist[nb] == -1:
                            dist[nb] = d + 1
                            prev[nb] = node
                            queue.append((nb, node))
                        else:
                            # Collision!
                            # If nb is the parent, it's backtracking, ignore.
                            if nb == p:
                                continue
                            # If we found a visited node, it's a cycle.
                            # Cycle length = dist[node] + dist[nb] + 1
                            # We need to ensure this length is >= g
                            # Actually, if dist[nb] <= d, the cycle is dist[node] + dist[nb] + 1
                            # If dist[nb] > d, it would have been found earlier?
                            # In BFS, if we encounter a visited node (not parent), it's a cycle.
                            # The shortest cycle through this edge is dist[node] + dist[nb] + 1.
                            # If this is < g, invalid.
                            if dist[node] + dist[nb] + 1 < g:
                                valid = False
                                break
                if not valid:
                    break
            
            if valid:
                # Construct edges
                edges = []
                for i in range(n):
                    edges.append((i, (i + 1) % n))
                    edges.append((i, (i - 1) % n))
                    edges.append((i, (i + m) % n))
                # Remove duplicates (i, j) and (j, i)
                # Since it's undirected, (i, i+1) is same as (i+1, i).
                # My generation adds both directions?
                # (i, i+1) and (i+1, i+2)...
                # (i, i-1) is (i, i-1). (i-1, i) is generated when start=i-1.
                # So I have duplicates.
                unique_edges = set()
                for u, v in edges:
                    if u > v:
                        u, v = v, u
                    unique_edges.add((u, v))
                best_n = n
                best_edges = list(unique_edges)
                break # Found valid for this n, try next n? No, n is minimized by loop order.
        
        if best_n != float('inf'):
            # Found a valid circulant with this n. 
            # Since we iterate n upwards, this is the smallest n for circulant.
            # But maybe a non-circulant is smaller? 
            # We don't have time to search non-circulants.
            # So we return this.
            break

    if best_n != float('inf'):
        return best_n, best_edges

    # Fallback: L-U-W construction (from seed)
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
