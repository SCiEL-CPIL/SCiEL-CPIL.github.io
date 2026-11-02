import numpy as np
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def check_girth(adj, n):
        """Check girth using BFS from each vertex, non-backtracking."""
        girth = float('inf')
        for s in range(n):
            # BFS with state (vertex, prev) to avoid backtracking
            dist = {}
            queue = deque()
            for nb in adj[s]:
                dist[(s, nb)] = 1
                queue.append((s, nb))
            while queue:
                prev, v = queue.popleft()
                d = dist[(prev, v)]
                for nb in adj[v]:
                    if nb == prev:
                        continue
                    if (v, nb) in dist:
                        cycle_len = d + 1
                        if cycle_len < girth:
                            girth = cycle_len
                    else:
                        dist[(v, nb)] = d + 1
                        queue.append((v, nb))
        return girth
    
    def random_k_regular(n, k):
        """Generate a random k-regular simple graph."""
        for attempt in range(1000):
            half_edges = []
            for i in range(n):
                for _ in range(k):
                    half_edges.append(i)
            random.shuffle(half_edges)
            adj = {i: [] for i in range(n)}
            valid = True
            for i in range(0, n * k, 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid and all(len(adj[i]) == k for i in range(n)):
                return adj
        return None
    
    def find_shortest_cycle(adj, n, max_len=None):
        """Find shortest cycle, return (length, cycle_nodes) or (inf, None)."""
        best = float('inf') if max_len is None else max_len
        best_cycle = None
        for s in range(n):
            dist = {}
            parent = {}
            queue = deque()
            for nb in adj[s]:
                dist[(s, nb)] = 1
                parent[(s, nb)] = (s, s)
                queue.append((s, nb))
            while queue:
                prev, v = queue.popleft()
                d = dist[(prev, v)]
                for nb in adj[v]:
                    if nb == prev:
                        continue
                    if (v, nb) in dist:
                        cycle_len = d + 1
                        if cycle_len < best:
                            best = cycle_len
                            best_cycle = (s, v, nb)  # s-v-...-nb-s
                    else:
                        dist[(v, nb)] = d + 1
                        parent[(v, nb)] = (prev, v)
                        queue.append((v, nb))
        return best, best_cycle
    
    def try_edge_swap(adj, n, k):
        """Try random edge swaps to improve girth."""
        for _ in range(200):
            # Pick two non-adjacent edges (a,b) and (c,d) where a!=c, a!=d, etc.
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            random.shuffle(edges)
            for (a, b) in edges:
                for (c, d) in edges:
                    if a == c or a == d or b == c or b == d:
                        continue
                    # Check if (a,c),(b,d) or (a,d),(b,c) are valid
                    if c not in adj[a] and d not in adj[b] and a != c and b != d:
                        # Try swap 1: (a,b),(c,d) -> (a,c),(b,d)
                        new_adj = {u: list(adj[u]) for u in range(n)}
                        new_adj[a].remove(b)
                        new_adj[b].remove(a)
                        new_adj[c].remove(d)
                        new_adj[d].remove(c)
                        new_adj[a].append(c)
                        new_adj[c].append(a)
                        new_adj[b].append(d)
                        new_adj[d].append(b)
                        # Check no duplicates
                        valid = True
                        for u in range(n):
                            if len(new_adj[u]) != len(set(new_adj[u])):
                                valid = False
                                break
                        if valid:
                            return new_adj
                    if c not in adj[a] and d not in adj[b] and a != d and b != c:
                        # Try swap 2: (a,b),(c,d) -> (a,d),(b,c)
                        new_adj = {u: list(adj[u]) for u in range(n)}
                        new_adj[a].remove(b)
                        new_adj[b].remove(a)
                        new_adj[c].remove(d)
                        new_adj[d].remove(c)
                        new_adj[a].append(d)
                        new_adj[d].append(a)
                        new_adj[b].append(c)
                        new_adj[c].append(b)
                        valid = True
                        for u in range(n):
                            if len(new_adj[u]) != len(set(new_adj[u])):
                                valid = False
                                break
                        if valid:
                            return new_adj
        return None
    
    def edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges
    
    # Try to find a 4-regular graph of girth >= 10
    best_n = None
    best_edges = None
    
    for n in range(102, 150):
        if time.time() > deadline - 20:
            break
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            adj = random_k_regular(n, k)
            if adj is None:
                continue
            # Try to improve girth
            improved = True
            while improved and time.time() < deadline - 20:
                improved = False
                girth_val, _ = find_shortest_cycle(adj, n, max_len=g)
                if girth_val >= g:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges_from_adj(adj, n)
                    break
                # Try edge swaps
                for _ in range(50):
                    new_adj = try_edge_swap(adj, n, k)
                    if new_adj is None:
                        break
                    new_girth, _ = find_shortest_cycle(new_adj, n, max_len=g)
                    if new_girth > girth_val:
                        adj = new_adj
                        improved = True
                        girth_val = new_girth
                        if girth_val >= g:
                            if best_n is None or n < best_n:
                                best_n = n
                                best_edges = edges_from_adj(adj, n)
                            break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: LUW construction
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    return luw_construct()
