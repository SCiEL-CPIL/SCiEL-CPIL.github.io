"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth >= g using randomized search with LUW fallback."""
    deadline = time.time() + 110
    
    # Moore bound
    moore = 2 + k * sum((k-1)**i for i in range(g // 2))
    
    def check_girth(adj, n, min_girth):
        """Check if graph has no cycle shorter than min_girth."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    def luw_construct(d, q):
        """LUW construction."""
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
    
    # Try randomized search for small n
    best_result = None
    
    # Try random k-regular graphs for n from moore to moore + 200
    for n in range(moore, min(moore + 200, 500), 2):
        if time.time() > deadline:
            break
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Generate random k-regular graph
            half_edges = list(range(n)) * k
            random.shuffle(half_edges)
            
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, 2 * n * k, 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if not valid:
                continue
            
            adj_list = [list(s) for s in adj]
            if check_girth(adj_list, n, g):
                edges = []
                for i in range(n):
                    for j in adj_list[i]:
                        if i < j:
                            edges.append((i, j))
                best_result = (n, edges)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = luw_construct(d, q)
    return n, edges
# EVOLVE-BLOCK-END
