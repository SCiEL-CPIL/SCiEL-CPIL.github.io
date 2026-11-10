"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    # Primary Strategy: Circulant Graph Search
    # Only effective for k=4
    if k == 4:
        # Moore bound estimate for k=4, g=10 is ~250
        # We search for n in increasing order
        # Girth check function
        def check_circulant_girth(n, S, g_target):
            # S is a set of steps
            dist = [-1] * n
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
                        # Back edge found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g_target:
                            return False
            return min_cycle >= g_target

        # Search range
        # Lower bound 243, try up to 350
        # If we find a graph, we return it.
        # We prioritize smaller n.
        
        # To speed up, we can pre-filter S sets?
        # Just random sampling is fine.
        
        for n in range(250, 351):
            if time.time() > deadline:
                break
            
            # Try to find a valid S for this n
            found = False
            # Number of random trials per n
            # 1000 trials is fast
            for _ in range(1000):
                # Pick a, b
                # a, b must be in 1..n-1
                # S = {a, -a, b, -b}
                # Ensure a != b, a != -b, etc.
                a = random.randint(1, n - 1)
                b = random.randint(1, n - 1)
                
                # Normalize to positive steps for uniqueness in set?
                # S is a set of steps. {a, n-a, b, n-b}
                # We need 4 distinct steps.
                # a != b, a != n-b, n-a != b, n-a != n-b
                # i.e., a != b, a + b != n, a != n-b (same as a+b!=n), a != b (same)
                # Also a != 0, b != 0.
                
                if a == b or a == n - b:
                    continue
                
                S = {a, n - a, b, n - b}
                if len(S) != 4:
                    continue
                
                if check_circulant_girth(n, S, g):
                    # Construct edges
                    edges = []
                    for i in range(n):
                        for s in S:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges
            # If not found in 1000 trials, try next n?
            # Or try more trials?
            # 1000 trials is likely enough to find one if it exists.
            # If it doesn't exist, we move on.

    # Fallback to LUW
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
