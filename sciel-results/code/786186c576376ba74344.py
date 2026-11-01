"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

def construct(k, g):
    deadline = time.time() + 145  # Leave 5s safety margin

    def check_girth_fast(adj, n, target_girth):
        """Check if graph has girth >= target_girth using non-backtracking BFS from each vertex."""
        max_depth = (target_girth - 1) // 2
        for start in range(n):
            visited = [[-1] * n for _ in range(max_depth + 1)]
            visited[0][start] = -2
            queue = deque([(start, 0, -1)])
            while queue:
                v, depth, parent = queue.popleft()
                if depth >= max_depth:
                    continue
                for u in adj[v]:
                    if u == parent:
                        continue
                    if visited[depth + 1][u] != -1:
                        return False
                    visited[depth + 1][u] = v
                    queue.append((u, depth + 1, v))
        return True

    def build_lift(base_edges, voltages, m, k):
        """Build the voltage graph lift. Returns (n, adj, edges) or None."""
        n_b = max(max(u, v) for u, v in base_edges) + 1
        n = n_b * m
        adj = [[] for _ in range(n)]
        
        for (u, v), vol in zip(base_edges, voltages):
            vol = vol % m
            for x in range(m):
                u_lift = u * m + x
                v_lift = v * m + ((x + vol) % m)
                adj[u_lift].append(v_lift)
        
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        for i in range(n):
            seen = set()
            for j in adj[i]:
                if j == i:
                    return None
                if j in seen:
                    return None
                seen.add(j)
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        return n, adj, edges

    def try_voltage_assignment(base_edges, m, k, g, deadline, rng):
        """Try random voltage assignments on the base graph."""
        num_edges = len(base_edges)
        
        for trial in range(50):
            if time.time() > deadline:
                return None
            
            voltages = [rng.randint(0, m - 1) for _ in range(num_edges)]
            result = build_lift(base_edges, voltages, m, k)
            if result is None:
                continue
            
            n, adj, edges = result
            
            if check_girth_fast(adj, n, g):
                return n, edges
        
        return None

    # K6 base graph for k=5
    base_edges_k6 = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    
    best_result = None
    rng = random.Random(42)
    
    # Strategy 1: K6 with various m values
    for m in range(60, 200, 2):
        if time.time() > deadline:
            break
        result = try_voltage_assignment(base_edges_k6, m, k, g, deadline, rng)
        if result is not None:
            n, edges = result
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
                if n < 500:
                    return n, edges
    
    # Strategy 2: K(5,5) base graph (10 vertices, 5-regular)
    base_edges_k55 = [(i, 5 + j) for i in range(5) for j in range(5)]
    
    for m in range(36, 150, 2):
        if time.time() > deadline:
            break
        result = try_voltage_assignment(base_edges_k55, m, k, g, deadline, rng)
        if result is not None:
            n, edges = result
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
                if n < 500:
                    return n, edges
    
    if best_result is not None:
        return best_result
    
    # Original LUW construction as fallback
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
# EVOLVE-BLOCK-END
