import time
import random
from collections import deque
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    # LUW Construction (Fallback)
    def luw_construction():
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

    n_luw, edges_luw = luw_construction()
    best_n, best_edges = n_luw, edges_luw

    # Heuristic Search for Smaller Graph
    def try_heuristic_search(n_target, k, g, deadline):
        if n_target < 242: # Moore bound
            return None
        
        def is_valid_graph(n, edges):
            # Check degree
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != k for d in deg):
                return False
            
            # Build Adj
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Check Girth
            # Non-backtracking BFS from each vertex
            for s in range(n):
                for a in adj[s]:
                    dist = {}
                    dist[(a, s)] = 1
                    q = deque([(a, s)])
                    while q:
                        v, p = q.popleft()
                        d = dist[(v, p)]
                        if d >= g - 1:
                            continue
                        for w in adj[v]:
                            if w == p:
                                continue
                            new_state = (w, v)
                            if new_state not in dist:
                                dist[new_state] = d + 1
                                if w == s and v != a:
                                    if d + 1 < g:
                                        return False
                                q.append(new_state)
            return True

        def random_k_regular(n, k, seed):
            rng = random.Random(seed)
            half_edges = []
            for i in range(n):
                for _ in range(k):
                    half_edges.append(i)
            rng.shuffle(half_edges)
            edges = set()
            adj = [[] for _ in range(n)]
            for i in range(0, n * k, 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v:
                    return None
                key = (min(u, v), max(u, v))
                if key in edges:
                    return None
                edges.add(key)
                adj[u].append(v)
                adj[v].append(u)
            return sorted([list(e) for e in edges]), adj

        def find_short_cycles(adj, n, g):
            cycles = []
            for s in range(n):
                for a in adj[s]:
                    dist = {}
                    dist[(a, s)] = 1
                    parent = {}
                    q = deque([(a, s)])
                    while q:
                        v, p = q.popleft()
                        d = dist[(v, p)]
                        if d >= g - 1:
                            continue
                        for w in adj[v]:
                            if w == p:
                                continue
                            new_state = (w, v)
                            if new_state not in dist:
                                dist[new_state] = d + 1
                                parent[new_state] = (v, p)
                                if w == s and v != a:
                                    if d + 1 < g:
                                        # Found a cycle
                                        # Reconstruct
                                        path = []
                                        cur = (w, v)
                                        while cur[0] != s:
                                            path.append(cur[0])
                                            cur = parent[cur]
                                        path.append(s)
                                        cycles.append(path)
                                q.append(new_state)
            return cycles

        def swap_edges(edges, adj, e1, e2):
            # e1, e2 are (u, v)
            u1, v1 = e1
            u2, v2 = e2
            # Check if swapping creates loops or multi-edges
            # Remove e1, e2
            edges.remove(e1)
            edges.remove(e2)
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            # Try all 3 pairings
            pairings = [
                ((u1, u2), (v1, v2)),
                ((u1, v2), (v1, u2)),
                ((u1, v1), (u2, v2)) # Original, skip
            ]
            # Actually, we remove and re-add.
            # We want to break a cycle.
            # Let's just try random swaps.
            
            # Revert
            edges.add(e1)
            edges.add(e2)
            adj[u1].append(v1)
            adj[v1].append(u1)
            adj[u2].append(v2)
            adj[v2].append(u2)
            return None # Placeholder

        # Simple Loop: Generate, Check, Return
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            edges, adj = random_k_regular(n_target, k, seed=attempt)
            if edges is None:
                continue
            if is_valid_graph(n_target, edges):
                return n_target, edges
        return None

    # Try 320
    if time.time() < deadline - 10:
        result = try_heuristic_search(320, k, g, deadline)
        if result:
            best_n, best_edges = result

    return best_n, best_edges
