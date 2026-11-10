import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        # For k=4, g=10, d=4 gives n=512
        # For k=4, g=8, d=3 gives n=128
        # We want girth >= g
        # Girth of LUW is approx 2*d + 2
        # So 2*d + 2 >= g => d >= (g-2)/2
        d = max(3, (g - 2) // 2)
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
        # Precompute rules
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
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # Primary Strategy: Circulant Graph Search
    if k == 4:
        def check_circulant_girth(n, S, g_target):
            # BFS from 0
            dist = np.full(n, -1, dtype=np.int32)
            dist[0] = 0
            queue = [0]
            min_cycle = float('inf')
            
            for u in queue:
                for s in S:
                    v = (u + s) % n
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        # Back edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g_target:
                            return False
            return min_cycle >= g_target

        # Search for n in range
        # Known cage is 320. Try 250-350.
        for n in range(250, 351):
            if time.time() > deadline:
                break
            
            # Try random sets of steps
            # S = {a, -a, b, -b}
            trials = 500
            for _ in range(trials):
                a = random.randint(1, n - 1)
                b = random.randint(1, n - 1)
                if a == b or a == n - b:
                    continue
                S = {a, n - a, b, n - b}
                if len(S) != 4:
                    continue
                
                if check_circulant_girth(n, S, g):
                    edges = []
                    for i in range(n):
                        for s in S:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    # Fallback to LUW
    return luw_construct(k, g)
