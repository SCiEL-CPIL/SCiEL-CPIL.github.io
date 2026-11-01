import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def _known_cages(k, g):
        """Return (n, edges) for known small cages, or None."""
        # Petersen graph: 3-regular, girth 5, n=10
        if k == 3 and g == 5:
            edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                     (5,7),(7,9),(9,6),(6,8),(8,5),
                     (0,5),(1,6),(2,7),(3,8),(4,9)]
            return 10, edges
        # Heawood graph: 3-regular, girth 7, n=14
        if k == 3 and g == 7:
            lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
            edges = []
            for i, line in enumerate(lines):
                for pt in line:
                    edges.append((pt, 7 + i))
            return 14, edges
        # (4,5)-cage: n=20
        if k == 4 and g == 5:
            # The (4,5)-cage is the 20-vertex graph.
            # One construction: The line graph of K_{5,5} is 8-regular. No.
            # The (4,5)-cage is the 4-regular graph on 20 vertices with girth 5.
            # It is the complement of the 15-vertex... no.
            # Actually, the (4,5)-cage is the 20-vertex graph formed by
            # taking two 5-cycles and connecting them? No.
            # I will rely on LUW for this if not found.
            return None
        return None

    def luw_construct(k, g, d):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        elif q in [2, 3, 5, 7, 11, 13, 17, 19]:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        else:
            return None
        
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
    
    def has_short_cycle(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Non-backtracking BFS: state is (current, prev)
        for start_u in range(n):
            for start_v in adj[start_u]:
                visited = set()
                visited.add((start_u, start_v))
                queue = deque()
                queue.append((start_v, start_u, 1))
                
                while queue:
                    curr, prev, depth = queue.popleft()
                    if depth >= g - 1:
                        continue
                    for nb in adj[curr]:
                        if nb == prev:
                            continue
                        state = (curr, nb)
                        if state in visited:
                            if depth + 1 < g:
                                return True
                        else:
                            visited.add(state)
                            queue.append((nb, curr, depth + 1))
        return False

    # Try known cages first
    known = _known_cages(k, g)
    if known is not None:
        return known

    # Calculate Moore Bound to determine minimum d
    # Moore bound for k-regular graph of girth g
    if g % 2 == 0:
        # g = 2r
        r = g // 2
        mb = 1 + k * sum((k-1)**i for i in range(r-1)) + (k-1)**(r-1) # Approx
        # Actually for even g=2r: 2 * (1 + k * sum_{i=0}^{r-2} (k-1)^i)
        mb = 2 * (1 + k * sum((k-1)**i for i in range(r-1)))
    else:
        # g = 2r+1
        r = (g - 1) // 2
        mb = 1 + k * sum((k-1)**i for i in range(r))

    # Find minimum d such that 2 * k^d >= mb
    d_min = 3
    while 2 * (k ** d_min) < mb:
        d_min += 1
        if d_min > 10: break # Safety

    # Determine the d used by the seed program
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)

    # Try d values from d_min to d_seed
    best_n = None
    best_edges = None
    
    for d in range(d_min, d_seed + 1):
        if time.time() > deadline:
            break
        result = luw_construct(k, g, d)
        if result is None:
            continue
        n, edges = result
        if not has_short_cycle(n, edges, g):
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
            # Since we are increasing d, n increases. 
            # If d_min works, it is the best.
            break

    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to d_seed if nothing worked (should not happen if seed is valid)
    result = luw_construct(k, g, d_seed)
    if result is not None:
        return result

    raise ValueError(f"Cannot construct {k}-regular graph of girth {g}")
