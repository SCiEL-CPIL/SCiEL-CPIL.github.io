import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construct(k, g, d_override=None, q_override=None, rule_override=None):
        q = q_override if q_override is not None else k
        if d_override is not None:
            d = d_override
        else:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if rule_override is not None:
            rule = rule_override
        else:
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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

    def verify_girth(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                v = queue.popleft()
                if dist[v] >= target_girth // 2:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and v != parent[w]:
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    if k != 3 or g != 16:
        # For general (k,g), try a few d values
        best_n = float('inf')
        best_edges = None
        for d in range(max(3, g - 6), g + 2):
            if time.time() > deadline:
                break
            n, edges = luw_construct(k, g, d_override=d)
            if n < best_n:
                if verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
        if best_edges is not None:
            return best_n, best_edges
        return luw_construct(k, g)

    # Specialized for (3, 16): search over rule variations and d
    best_n = float('inf')
    best_edges = None

    # Strategy 1: Try different d values with standard rule
    for d in range(6, 14):
        if time.time() > deadline:
            break
        n, edges = luw_construct(k, g, d_override=d)
        if n < best_n:
            if verify_girth(n, edges, g):
                best_n = n
                best_edges = edges

    # Strategy 2: Try different rule patterns for smaller d
    # The rule maps level c to (a, b) indices
    # Try systematic variations
    for d in range(5, 10):
        if time.time() > deadline:
            break
        # Try all rule variations where rule[c] = (a, b) with a, b in valid range
        # For level c, valid indices are 1..c-1 for the "other" side
        base_rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            base_rule[c] = (1, c - 1)  # default: use first and last
        
        # Try a few systematic rule patterns
        patterns = [
            lambda c: (1, c - 1),
            lambda c: (1, 1),
            lambda c: (c - 1, 1),
            lambda c: (1, c // 2),
            lambda c: (c // 2, 1),
        ]
        
        for pat_idx, pat in enumerate(patterns):
            if time.time() > deadline:
                break
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                rule[c] = pat(c)
            # Ensure indices are valid: a, b should be in [1, c-1]
            valid = True
            for c in range(5, d + 1):
                a, b = rule[c]
                if not (1 <= a <= c - 1 and 1 <= b <= c - 1):
                    valid = False
                    break
            if not valid:
                continue
            n, edges = luw_construct(k, g, d_override=d, rule_override=rule)
            if n < best_n:
                if verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges

    # Strategy 3: Randomized rule search
    random.seed(42)
    for d in range(5, 12):
        if time.time() > deadline:
            break
        max_attempts = max(1, int((deadline - time.time()) / 0.5))
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            valid = True
            for c in range(5, d + 1):
                a = random.randint(1, c - 1)
                b = random.randint(1, c - 1)
                rule[c] = (a, b)
            n, edges = luw_construct(k, g, d_override=d, rule_override=rule)
            if n < best_n:
                if verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    # Fallback
    return luw_construct(k, g)
