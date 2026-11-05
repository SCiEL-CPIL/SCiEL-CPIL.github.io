import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target n: Try to find a graph smaller than the LUW component.
    # LUW component for q=3, d=11 is likely ~4374 or 1080.
    # Known best is 936.
    # We will try to construct a graph of size 936.
    
    target_n = 936
    if target_n % 2 != 0:
        target_n += 1
        
    # Function to generate a random 3-regular graph
    def gen_random_3reg(n):
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edges = []
        used = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                continue
            if (u, v) in used or (v, u) in used:
                continue
            edges.append((u, v))
            used.add((u, v))
            used.add((v, u))
        # Check if 3-regular
        deg = [0]*n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            return None
        return edges

    # Function to check girth
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            # BFS up to depth (min_girth - 1) // 2
            # Actually, to detect cycle of length L, we need to find a path of length L-1 that closes.
            # If we find a cycle of length < min_girth, return False.
            visited = {start: 0}
            queue = [(start, -1, 0)]
            while queue:
                u, parent, dist = queue.pop(0)
                if dist >= (min_girth - 1) // 2:
                    continue
                for v in adj[u]:
                    if v == parent:
                        continue
                    if v in visited:
                        # Found a cycle
                        cycle_len = dist + visited[v] + 1
                        if cycle_len < min_girth:
                            return False
                    else:
                        visited[v] = dist + 1
                        queue.append((v, u, dist + 1))
        return True

    # Try to find a valid graph
    best_n = float('inf')
    best_edges = None
    
    # Try a few random restarts for n=936
    for _ in range(50):
        if time.time() > deadline:
            break
        edges = gen_random_3reg(target_n)
        if edges is None:
            continue
        if check_girth(target_n, edges, g):
            best_n = target_n
            best_edges = edges
            break
        # If not valid, we could try local search, but it's complex.
        # Just try another random graph.
        
    # If we found one, return it
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback: LUW construction (same as seed)
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
