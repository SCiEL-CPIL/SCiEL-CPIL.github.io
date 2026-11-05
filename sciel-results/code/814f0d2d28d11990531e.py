"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 145.0  # safety margin before 150s kill

    q = k  # q must be prime power; for k=4, q=4 (GF(4))

    # GF(q) operations
    if q == 4:
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    def build_luw(d):
        """Build the connected component of LUW graph CD(d, q) from vertex 0."""
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                a, b = 1, c0 - 2
            elif r == 1:
                a, b = c0 - 1, 1
            elif r == 2:
                a, b = c0, 1
            else:
                a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b

        mul_lookup = [[mul(i, j) for j in range(q)] for i in range(q)]

        def compute_neighbours(vertex):
            side, x = vertex
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add(x[c], mul_lookup[y[a]][x[b]])
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub(x[c], mul_lookup[x[a]][y[b]])
                result.append((1 - side, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []

        head = 0
        while head < len(order):
            if time.time() > deadline:
                break
            vertex = order[head]
            head += 1

            for other in compute_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        n = len(order)
        return n, edges

    def has_short_cycle(n, edges, max_len):
        """Check if graph has any cycle of length < max_len using non-backtracking walks.
        Returns True if a short cycle exists (bad), False if girth >= max_len."""
        if n <= 1:
            return False

        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return True  # invalid

        # Non-backtracking BFS: state = (vertex, prev)
        # A cycle of length L exists if we can reach (start, some_prev) 
        # where some_prev != the first vertex we came from, after L steps
        # Simpler: for each directed edge (u,v), do BFS with no backtracking
        # and check if we can reach (v,u) in fewer than max_len-1 steps

        # More efficient: for each vertex, do BFS tracking (v, prev) 
        # and check if we return to start with a different prev

        for start in range(n):
            # BFS from each neighbor of start
            for first_nb in adj[start]:
                visited = set()
                visited.add((first_nb, start))
                queue = deque([(first_nb, start, 1)])

                while queue:
                    if time.time() > deadline:
                        return False  # assume no short cycle if running out of time
                    v, prev, dist = queue.popleft()

                    if dist >= max_len - 1:
                        continue

                    for nb in adj[v]:
                        if nb == prev:
                            continue
                        if nb == start and dist + 1 < max_len:
                            return True  # found cycle of length dist+1 < max_len
                        new_state = (nb, v)
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append((nb, v, dist + 1))

        return False

    # Compute minimal d with guaranteed girth >= g
    d_min = 3
    while True:
        girth_est = d_min + 5 if d_min % 2 == 1 else d_min + 4
        if girth_est >= g:
            break
        d_min += 1

    best_result = None

    # Try d from d_min-2 to d_min+2, smallest first
    for d in range(max(3, d_min - 2), d_min + 3):
        if time.time() > deadline - 10:  # leave time for verification
            break
        n, edges = build_luw(d)
        if n == 0 or len(edges) != n * k // 2:
            continue

        # Verify no cycle shorter than g
        if not has_short_cycle(n, edges, g):
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)

    # Fallback
    if best_result is None:
        n, edges = build_luw(d_min)
        best_result = (n, edges)

    return best_result
# EVOLVE-BLOCK-END
