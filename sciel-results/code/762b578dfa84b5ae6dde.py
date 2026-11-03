import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # 1. Attempt Cyclic Construction for k=4
    if k == 4:
        R = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(R))
        
        # Search n in [moore_bound, moore_bound + 500]
        # If a cyclic graph exists, it's likely close to the Moore bound.
        search_limit = moore_bound + 500
        
        for n in range(moore_bound, search_limit + 1):
            if time.time() > deadline - 10:
                break
                
            half = n // 2
            # Generate valid pairs (a, b)
            # a, b in [1, half]
            # a != b, a != -b mod n, a != n/2, b != n/2
            valid_pairs = []
            for a in range(1, half):
                if a == n / 2:
                    continue
                for b in range(a + 1, half):
                    if b == n / 2:
                        continue
                    # Check if a and b are distinct mod n and not negatives
                    if (a + b) % n == 0:
                        continue
                    if (a - b) % n == 0:
                        continue
                    valid_pairs.append((a, b))
            
            # Shuffle to find a solution faster if not the first one
            random.shuffle(valid_pairs)
            
            found = False
            for a, b in valid_pairs:
                if time.time() > deadline - 5:
                    break
                
                # Check girth using non-backtracking BFS
                steps = [a, -a, b, -b]
                # State: (residue, last_step_index)
                visited = set()
                queue = deque()
                
                # Initialize
                for i, s in enumerate(steps):
                    r = s % n
                    visited.add((r, i))
                    queue.append((r, i, 1))
                
                short_cycle = False
                while queue:
                    r, last_idx, depth = queue.popleft()
                    if depth >= g:
                        continue
                    for i, s in enumerate(steps):
                        # Non-backtracking: i != (last_idx + 2) % 4
                        if i == (last_idx + 2) % 4:
                            continue
                        nr = (r + s) % n
                        new_depth = depth + 1
                        if (nr, i) not in visited:
                            visited.add((nr, i))
                            queue.append((nr, i, new_depth))
                        
                        # Check if we returned to start (0)
                        if nr == 0 and new_depth < g:
                            short_cycle = True
                            break
                    if short_cycle:
                        break
                
                if not short_cycle:
                    # Valid graph found
                    edges = set()
                    for i in range(n):
                        j1 = (i + a) % n
                        edges.add((min(i, j1), max(i, j1)))
                        j2 = (i + b) % n
                        edges.add((min(i, j2), max(i, j2)))
                    return n, list(edges)

    # 2. Fallback to LUW Construction
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

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
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    queue = [0]
    head = 0

    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))

    return len(order), edges
# EVOLVE-BLOCK-END
