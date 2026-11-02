import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    def luw_construct(q, d):
        table = None
        if q == 4:
            pass
            
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
        
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

    def check_girth(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for u in range(n):
            visited = [-1] * n
            parent = [-1] * n
            depth = [0] * n
            visited[u] = 0
            queue = [u]
            
            while queue:
                curr = queue.pop(0)
                for nb in adj[curr]:
                    if visited[nb] == -1:
                        visited[nb] = 1
                        parent[nb] = curr
                        depth[nb] = depth[curr] + 1
                        if depth[nb] < target_girth:
                            queue.append(nb)
                    else:
                        if nb == parent[curr]:
                            continue
                        cycle_len = depth[curr] + depth[nb] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    def random_k_regular(n, k):
        # Wormald-Haggstrom algorithm
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Simple greedy matching with backtracking if needed, but for large n, simple shuffle usually works or fails.
        # For robustness, use a simpler method for small n, or accept failure.
        # Actually, for n=500, simple shuffle might create multi-edges.
        # We'll use a retry loop.
        
        while True:
            adj = [set() for _ in range(n)]
            edges = []
            success = True
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    success = False
                    break
                if v in adj[u]:
                    success = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
            
            if success:
                return adj, edges

    def find_short_cycle(adj, max_len):
        n = len(adj)
        for u in range(n):
            visited = [-1] * n
            parent = [-1] * n
            depth = [0] * n
            visited[u] = 0
            queue = [u]
            
            while queue:
                curr = queue.pop(0)
                for nb in adj[curr]:
                    if visited[nb] == -1:
                        visited[nb] = 1
                        parent[nb] = curr
                        depth[nb] = depth[curr] + 1
                        if depth[nb] < max_len:
                            queue.append(nb)
                    else:
                        if nb == parent[curr]:
                            continue
                        cycle_len = depth[curr] + depth[nb] + 1
                        if cycle_len < max_len:
                            # Reconstruct cycle
                            path = [curr]
                            node = curr
                            while node != nb:
                                node = parent[node]
                                path.append(node)
                            path.append(nb)
                            return path
        return None

    def switch_edges(adj, edges, u, v, x, y):
        # Remove (u,v) and (x,y), add (u,x) and (v,y)
        # Check validity
        if u == x or u == y or v == x or v == y:
            return False
        if x in adj[u] or y in adj[u]:
            return False
        if x in adj[v] or y in adj[v]:
            return False
            
        # Update
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].add(x)
        adj[x].add(u)
        adj[v].add(y)
        adj[y].add(v)
        
        # Update edges list
        try:
            edges.remove((u, v))
        except ValueError:
            edges.remove((v, u))
        try:
            edges.remove((x, y))
        except ValueError:
            edges.remove((y, x))
        edges.append((u, x))
        edges.append((v, y))
        return True

    def edge_switching(adj, edges, target_girth, max_steps=10000):
        n = len(adj)
        for _ in range(max_steps):
            if time.time() > deadline:
                return False
            cycle = find_short_cycle(adj, target_girth)
            if not cycle:
                return True
            # Pick two edges from the cycle
            # Cycle nodes: c[0], c[1], ..., c[L-1], c[0]
            # Edges: (c[0], c[1]), (c[1], c[2]), ...
            L = len(cycle)
            if L < 3:
                continue
            # Randomly pick two non-adjacent edges?
            # Just pick two random edges from the cycle
            e1_idx = random.randint(0, L-1)
            e2_idx = random.randint(0, L-1)
            if e1_idx == e2_idx:
                continue
                
            u1, v1 = cycle[e1_idx], cycle[(e1_idx+1)%L]
            u2, v2 = cycle[e2_idx], cycle[(e2_idx+1)%L]
            
            # Try switch
            if switch_edges(adj, edges, u1, v1, u2, v2):
                continue
            else:
                # Try other pairing? (u1, v2) and (u2, v1)
                if switch_edges(adj, edges, u1, v2, u2, v1):
                    continue
            # If both fail, try another pair
        return False

    # Try edge switching for smaller n
    for n in [500, 1000, 2000]:
        if time.time() > deadline:
            break
        for attempt in range(10):
            if time.time() > deadline:
                break
            adj, edges = random_k_regular(n, k)
            if edge_switching(adj, edges, g, max_steps=5000):
                if check_girth(n, edges, g):
                    return n, edges

    # Fallback: LUW
    if k == 5:
        d = 5
        n, edges = luw_construct(5, d)
        if check_girth(n, edges, g):
            return n, edges
            
    return n, edges
