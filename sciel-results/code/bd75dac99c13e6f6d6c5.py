"""
Construct a 5-regular graph of girth 9 (cage problem n(5,9)).

Strategy:
1. Use the Lazebnik-Ustimenko-Woldar (LUW) construction as a deterministic baseline.
2. Attempt to find a smaller valid graph via randomized 2-switch optimization
   starting from random regular graphs near the Moore bound.
3. Return the best valid graph found within the time budget.
"""

import time
import random
from collections import deque


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill

    # ------------------------------------------------------------------
    # Part 1: LUW construction (deterministic, always valid)
    # ------------------------------------------------------------------
    def luw_construction():
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

    # Get baseline immediately
    n_luw, edges_luw = luw_construction()
    best_n = n_luw
    best_edges = edges_luw

    # ------------------------------------------------------------------
    # Part 2: Randomized 2-switch optimization
    # ------------------------------------------------------------------
    def girth_check(n, adj, g):
        """Return True if girth >= g using BFS from each vertex."""
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    def gen_random_k_reg(n, k):
        """Generate a random simple k-regular graph via configuration model + repair."""
        for attempt in range(200):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid and all(len(a) == k for a in adj):
                return adj
        return None

    def count_short_cycles(n, adj, g):
        """Count cycles of length < g (approximate: triangles + 4-cycles)."""
        count = 0
        # Triangles
        for u in range(n):
            for i, v in enumerate(adj[u]):
                if u < v:
                    for w in adj[u][i+1:]:
                        if w in adj[v]:
                            count += 1
        # 4-cycles (sampled)
        if g > 4:
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        for w in adj[v]:
                            if w != u and w > v:
                                for x in adj[w]:
                                    if x != v and x > w and x in adj[u]:
                                        count += 1
        return count

    def two_switch(adj, n, k):
        """Perform a random 2-switch. Returns True if successful."""
        u = random.randint(0, n - 1)
        if len(adj[u]) < 2:
            return False
        v = random.choice(adj[u])
        w = random.randint(0, n - 1)
        if w == u or w == v:
            return False
        if len(adj[w]) < 2:
            return False
        x = random.choice(adj[w])
        if x == u or x == v or x == w:
            return False
        # Check no multi-edge
        if w in adj[u] or x in adj[v] or w == x:
            return False
        # Perform swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[w].remove(x)
        adj[x].remove(w)
        adj[u].append(w)
        adj[w].append(u)
        adj[v].append(x)
        adj[x].append(v)
        return True

    def optimize_from_adj(adj, n, k, g, time_limit):
        """Local search via 2-switches to eliminate short cycles."""
        end_time = time.time() + time_limit
        current_penalty = count_short_cycles(n, adj, g)
        best_penalty = current_penalty
        best_adj = [list(a) for a in adj]

        while time.time() < end_time:
            if current_penalty == 0:
                if girth_check(n, adj, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return (n, edges)
                break

            # Try a 2-switch
            old_adj = [list(a) for a in adj]
            if two_switch(adj, n, k):
                new_penalty = count_short_cycles(n, adj, g)
                if new_penalty <= current_penalty:
                    current_penalty = new_penalty
                    if new_penalty < best_penalty:
                        best_penalty = new_penalty
                        best_adj = [list(a) for a in adj]
                else:
                    # Revert
                    adj = old_adj
            else:
                continue

        return None

    # Moore bound for reference
    moore_bound = 2 * (k ** ((g // 2) - 1) + k ** (g // 2) - 1) if g % 2 == 0 else \
                  k * (k - 1) ** ((g - 3) // 2) + 2 * (k - 1) ** ((g - 3) // 2)
    if g % 2 == 0:
        moore_bound = 2 * ((k ** (g // 2)) - (k - 1))
    else:
        moore_bound = k * ((k - 1) ** ((g - 1) // 2))

    # Try several sizes from Moore bound upward
    candidate_ns = []
    base = max(100, int(moore_bound * 0.8))
    for mult in [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
        n_try = max(50, int(base * mult))
        if n_try % 2 == 0 and n_try * k % 2 == 0:
            candidate_ns.append(n_try)
    # Also try fixed values
    for nv in [500, 600, 800, 1000, 1100, 1200]:
        if nv * k % 2 == 0 and nv not in candidate_ns:
            candidate_ns.append(nv)
    candidate_ns.sort()

    for n_try in candidate_ns:
        if time.time() > deadline - 10:
            break
        time_budget = min(20, deadline - time.time() - 5)
        if time_budget < 5:
            break

        adj = gen_random_k_reg(n_try, k)
        if adj is None:
            continue

        result = optimize_from_adj(adj, n_try, k, g, time_budget)
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found
            break

    # Final fallback: return LUW result if nothing better
    if best_n == n_luw:
        return n_luw, edges_luw
    else:
        return best_n, best_edges
