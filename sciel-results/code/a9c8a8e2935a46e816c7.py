import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

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

    def nonbacktracking_girth(adj, g):
        n = len(adj)
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
                                    return True
                            q.append(new_state)
        return False

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def is_valid_graph(n, edges, k, g):
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for i, j in edges:
            if i == j:
                return False
            if i < 0 or i >= n or j < 0 or j >= n:
                return False
            key = (min(i, j), max(i, j))
            if key in seen:
                return False
            seen.add(key)
            adj[i].append(j)
            adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        if nonbacktracking_girth(adj, g):
            return False
        return True

    def try_voltage_graph(k, g, deadline):
        # Specific for k=4, g=10
        if k != 4 or g != 10:
            return None
        
        # Base graph: Bipartite double cover of K5
        # Vertices 0..4 (top), 5..9 (bottom)
        base_edges = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    base_edges.append((i, 5 + j))
        
        n_lift = 320
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random voltages in Z_32
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, 31)
            
            # Build the lift
            edges = []
            for u, v in base_edges:
                w = voltages[(u, v)]
                for x in range(32):
                    u_id = u * 32 + x
                    v_id = v * 32 + (x + w) % 32
                    edges.append((u_id, v_id))
            
            # Check validity
            if is_valid_graph(n_lift, edges, k, g):
                return n_lift, edges
        
        return None

    # First, get the LUW construction as baseline
    n_luw, edges_luw = luw_construction()
    best_n, best_edges = n_luw, edges_luw

    # Try voltage graph construction for specific case
    result = try_voltage_graph(k, g, deadline)
    if result is not None:
        n_found, edges_found = result
        if n_found < best_n:
            best_n, best_edges = n_found, edges_found

    return best_n, best_edges
