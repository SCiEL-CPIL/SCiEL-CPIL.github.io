import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Fallback: LUW Graph
    def luw_construct():
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

    # Try to find a Circulant Graph
    # Moore bound for k=5, g=9 is 426.
    # We search n from 426 to 1500.
    # For each n, we try random sets of 5 generators.
    
    def check_circulant(n, S):
        # S is a set of 5 generators in Z_n
        # Check for cycles of length < 9
        # BFS from 0
        # dist[v] = distance from 0
        dist = [-1] * n
        dist[0] = 0
        queue = [0]
        head = 0
        max_depth = 4 # We only need to check up to depth 4 to detect cycles < 9
        
        while head < len(queue) and dist[queue[head]] < max_depth:
            u = queue[head]
            head += 1
            du = dist[u]
            for s in S:
                v1 = (u + s) % n
                v2 = (u - s) % n
                for v in (v1, v2):
                    if dist[v] == -1:
                        dist[v] = du + 1
                        queue.append(v)
                    else:
                        # Found an edge (u, v)
                        # Cycle length is at most dist[u] + dist[v] + 1
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    best_n, best_edges = None, None
    
    # Search range
    # Start from Moore bound
    moore_bound = 1 + k * ((k - 1) ** ((g - 1) // 2) - 1) // (k - 2)
    
    # We might not find a circulant, so we limit the search time.
    # Try a few n values.
    for n in range(moore_bound, 1500):
        if time.time() > deadline - 10:
            break
        
        # Try random generators
        found = False
        # Number of trials per n
        trials = 500
        for _ in range(trials):
            if time.time() > deadline - 10:
                break
            
            # Pick 5 distinct non-zero generators
            # Ensure no s and -s both in S (to avoid 2-cycles/multi-edges)
            # Since graph is simple, we just need distinct generators.
            # But if s and n-s are both in S, it's the same edge.
            # So we pick from 1..n//2.
            half = n // 2
            if half < 5:
                continue
            
            S = set()
            while len(S) < 5:
                S.add(random.randint(1, half))
            S = list(S)
            
            # Check girth
            if check_circulant(n, S):
                # Construct edges
                edges = []
                for u in range(n):
                    for s in S:
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    # If no circulant found, return LUW
    return luw_construct()
