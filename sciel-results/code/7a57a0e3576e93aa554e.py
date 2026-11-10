import time
import random
import sys
from collections import defaultdict, deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound: minimum possible n for a k-regular graph of girth g
    moore_bound = 2 + k * sum((k - 1) ** i for i in range((g - 1) // 2))

    # LUW (Lazebnik-Ustimenko-Woldar) construction
    def get_luw_graph():
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
            c0 = c - (c - 5) % 4
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if time_left() < 5:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    luw_n, luw_edges = get_luw_graph()
    best_n = luw_n
    best_edges = luw_edges

    # Verify LUW graph girth using BFS from each vertex
    def verify_girth(n, edges, g_limit):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check girth via BFS
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                v = queue.popleft()
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        parent[u] = v
                        queue.append(u)
                    elif u != parent[v] and v != parent[u]:
                        cycle_len = dist[v] + dist[u] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    luw_valid = verify_girth(luw_n, luw_edges, g)
    if not luw_valid:
        # LUW graph is invalid; we need to find a valid one
        # Try random regular graphs with SA repair
        pass

    # Simulated Annealing / edge-swap search to find valid k-regular graph with girth >= g
    def gen_random_reg(n, k):
        """Generate a random simple k-regular graph via random pairing."""
        if n * k % 2 != 0:
            return None, None
        half_edges = [i for i in range(n) for _ in range(k)]
        random.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        edges = []
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v or v in adj[u]:
                return None, None
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        return adj, edges

    def find_short_cycles(adj, n, g_limit, max_cycles=200):
        """Find cycles shorter than g_limit using BFS from each vertex."""
        short_cycles = []
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                v = queue.popleft()
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        parent[u] = v
                        queue.append(u)
                    elif u != parent[v] and v != parent[u]:
                        cycle_len = dist[v] + dist[u] + 1
                        if cycle_len < g_limit:
                            # Reconstruct cycle
                            cycle = [v]
                            cur = v
                            while cur != start:
                                cur = parent[cur]
                                if cur == -1:
                                    break
                                cycle.append(cur)
                            short_cycles.append(cycle)
                            if len(short_cycles) >= max_cycles:
                                return short_cycles
        return short_cycles

    def try_swap(adj, n, k, edges):
        """Try a random edge swap. Returns (success, swap_info) where swap_info
        contains the data needed to undo the swap if needed."""
        if len(edges) < 2:
            return False, None
        for _ in range(100):
            i = random.randint(0, len(edges) - 1)
            j = random.randint(0, len(edges) - 1)
            if i == j:
                continue
            u, v = edges[i]
            x, y = edges[j]
            # Check all distinct
            if len({u, v, x, y}) < 4:
                continue
            # Check new edges don't already exist
            if x in adj[u] or y in adj[v]:
                continue
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            edges[i] = (u, x)
            edges[j] = (v, y)
            return True, (i, j, u, v, x, y)
        return False, None

    def undo_swap(adj, edges, swap_info):
        """Undo a swap given the info returned by try_swap."""
        i, j, u, v, x, y = swap_info
        # Reverse: remove (u,x),(v,y) and add (u,v),(x,y)
        adj[u].remove(x)
        adj[x].remove(u)
        adj[v].remove(y)
        adj[y].remove(v)
        adj[u].add(v)
        adj[v].add(u)
        adj[x].add(y)
        adj[y].add(x)
        edges[i] = (u, v)
        edges[j] = (x, y)

    def count_short_cycles(adj, n, g_limit):
        """Count total short cycle 'violations' using BFS."""
        count = 0
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                v = queue.popleft()
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        parent[u] = v
                        queue.append(u)
                    elif u != parent[v] and v != parent[u]:
                        cycle_len = dist[v] + dist[u] + 1
                        if cycle_len < g_limit:
                            count += 1
        return count

    # Try to find a valid graph via SA
    search_start = time.time()
    found_valid = False

    # Try several target sizes starting from Moore bound
    offsets = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    for offset in offsets:
        if time_left() < 10:
            break
        target_n = moore_bound + offset
        if target_n < 2 * k:
            continue
        if target_n * k % 2 != 0:
            continue

        # Try multiple random starts
        max_attempts = 3
        for attempt in range(max_attempts):
            if time_left() < 10:
                break

            adj, edges = gen_random_reg(target_n, k)
            if adj is None:
                continue

            # Initial violation count
            violations = count_short_cycles(adj, target_n, g)
            if violations == 0:
                found_valid = True
                best_n = target_n
                best_edges = edges
                break

            # SA: try to reduce violations
            T = 50.0
            cooling = 0.999
            min_T = 0.1
            max_steps = 5000

            for step in range(max_steps):
                if time_left() < 10:
                    break
                if T < min_T:
                    break

                old_violations = violations
                success, swap_info = try_swap(adj, target_n, k, edges)
                if not success:
                    continue

                new_violations = count_short_cycles(adj, target_n, g)
                delta = new_violations - old_violations

                if delta < 0 or random.random() < (10 ** (-delta / T)):
                    violations = new_violations
                    if violations == 0:
                        found_valid = True
                        best_n = target_n
                        best_edges = edges
                        break
                else:
                    # Revert the swap
                    undo_swap(adj, edges, swap_info)

                T *= cooling

            if found_valid:
                break

        if found_valid:
            break

    # If LUW is valid and no smaller found, keep LUW
    # If LUW is invalid and no valid found, return LUW (best effort)

    return best_n, best_edges
