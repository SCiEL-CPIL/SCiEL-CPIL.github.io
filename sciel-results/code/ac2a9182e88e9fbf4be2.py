"""
Construct a 5-regular graph of girth 9 (cage problem n(5,9)).

Strategy:
1. Use the Lazebnik-Ustimenko-Woldar (LUW) construction with optimized
   parameters to get a valid, k-regular, girth >= g graph.
2. Use proper non-backtracking walk girth verification.
3. Attempt local improvement via targeted 2-switches with correct girth checking.
"""

import time
import random
from collections import deque


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def check_girth_nonbacktracking(n, adj, g):
        """Check girth >= g using non-backtracking walks from each vertex.
        A non-backtracking walk of length L starting at vertex s with first step to v
        is a sequence s, v, w1, w2, ... where wi != v_{i-1}.
        We detect cycles of length < g by checking if any non-backtracking walk
        of length < g returns to its start."""
        for s in range(n):
            # State: (current_vertex, previous_vertex, length)
            # Start with all neighbors of s
            visited = set()
            for v in adj[s]:
                visited.add((v, s))
            queue = deque([(v, s, 1) for v in adj[s]])
            while queue:
                u, prev, length = queue.popleft()
                if length >= g:
                    continue
                for w in adj[u]:
                    if w == prev:
                        continue
                    if w == s and length + 1 < g:
                        return False
                    new_state = (w, u)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((w, u, length + 1))
        return True

    def luw_construction(q, d):
        """LUW construction with given parameters q and d."""
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
        return len(order), edges, order, index

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Try different LUW parameters
    best_n = None
    best_edges = None
    best_adj = None

    # For k=5, g=9: try various (q, d) combinations
    # d should be at least ceil((g-1)/2) for the construction to potentially have girth >= g
    # Actually the LUW construction's girth depends on the specific rule set
    candidate_params = []
    for q in [k]:
        for d in range(4, 10):
            candidate_params.append((q, d))

    for q, d in candidate_params:
        if time.time() > deadline - 30:
            break
        try:
            n, edges, order, index = luw_construction(q, d)
            adj = build_adj(n, edges)
            # Verify
            if all(len(adj[i]) == k for i in range(n)):
                if check_girth_nonbacktracking(n, adj, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                        best_adj = adj
        except Exception:
            continue

    # If no valid LUW graph found, use a larger d
    if best_n is None:
        for d in range(10, 15):
            if time.time() > deadline - 20:
                break
            try:
                n, edges, order, index = luw_construction(k, d)
                adj = build_adj(n, edges)
                if all(len(adj[i]) == k for i in range(n)):
                    if check_girth_nonbacktracking(n, adj, g):
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                            best_adj = adj
                        break
            except Exception:
                continue

    if best_n is None:
        # Last resort: use a very large d
        n, edges, order, index = luw_construction(k, 12)
        best_n = n
        best_edges = edges
        best_adj = build_adj(n, edges)

    # Try local optimization if time permits
    if best_adj is not None and time.time() < deadline - 15:
        def count_short_nonbacktracking_cycles(n, adj, g):
            """Count non-backtracking cycles of length < g."""
            count = 0
            for s in range(n):
                visited = set()
                for v in adj[s]:
                    visited.add((v, s))
                queue = deque([(v, s, 1) for v in adj[s]])
                while queue:
                    u, prev, length = queue.popleft()
                    if length >= g:
                        continue
                    for w in adj[u]:
                        if w == prev:
                            continue
                        if w == s and length + 1 < g:
                            count += 1
                        else:
                            new_state = (w, u)
                            if new_state not in visited:
                                visited.add(new_state)
                                queue.append((w, u, length + 1))
            return count

        def two_switch(adj, n):
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
            if w in adj[u] or x in adj[v] or w == x:
                return False
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            return True

        current_adj = [list(a) for a in best_adj]
        current_penalty = count_short_nonbacktracking_cycles(best_n, current_adj, g)
        
        while time.time() < deadline - 10 and current_penalty > 0:
            old_adj = [list(a) for a in current_adj]
            if two_switch(current_adj, best_n):
                new_penalty = count_short_nonbacktracking_cycles(best_n, current_adj, g)
                if new_penalty < current_penalty:
                    current_penalty = new_penalty
                else:
                    current_adj = old_adj
            else:
                current_adj = old_adj
        
        # Check if improved
        if current_penalty == 0 and check_girth_nonbacktracking(best_n, current_adj, g):
            edges = []
            for i in range(best_n):
                for j in current_adj[i]:
                    if i < j:
                        edges.append((i, j))
            best_edges = edges
            best_adj = current_adj

    return best_n, best_edges
