import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def check_girth_cyclic_fast(n, a, b):
        """
        Check if the cyclic graph defined by offsets a, b has girth >= g.
        """
        offs = [a, (-a) % n, b, (-b) % n]
        # Check all 4 offsets are distinct
        if len(set(offs)) != 4:
            return False
            
        visited = [[False] * 4 for _ in range(n)]
        queue = deque()
        
        for i in range(4):
            s = offs[i]
            visited[s][i] = True
            queue.append((s, i, 1))
            
        while queue:
            s, last_idx, depth = queue.popleft()
            
            for j in range(4):
                if j == (last_idx ^ 1):
                    continue
                    
                new_s = (s + offs[j]) % n
                
                if new_s == 0:
                    if depth + 1 < g:
                        return False
                    continue

                if depth + 1 < g and not visited[new_s][j]:
                    visited[new_s][j] = True
                    queue.append((new_s, j, depth + 1))
        
        return True

    def make_cyclic_graph(n, a, b):
        edges = []
        edge_set = set()
        for i in range(n):
            for off in [a, b]:
                for j in [(i + off) % n, (i - off) % n]:
                    if i == j:
                        return None
                    key = (min(i, j), max(i, j))
                    if key in edge_set:
                        return None
                    edge_set.add(key)
                    edges.append((i, j))
        return edges

    # Try cyclic graph search (known: no solution for (4,10) below 320, but try anyway for small n)
    if not (k == 4 and g == 10):
        for n in range(243, 401):
            if time.time() > deadline - 10:
                break
            half_n = n // 2
            for a in range(1, half_n + 1):
                if 2 * a % n == 0:
                    continue
                for b in range(a + 1, half_n + 1):
                    if 2 * b % n == 0:
                        continue
                    if (a + b) % n == 0:
                        continue
                    if (a - b) % n == 0:
                        continue
                    if check_girth_cyclic_fast(n, a, b):
                        edges = make_cyclic_graph(n, a, b)
                        if edges is not None:
                            return n, edges

    # Fallback: LW Graph
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
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
