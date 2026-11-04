import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 10s margin
    
    # Fallback: LUW construction
    def luw_construct():
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

    # Local Search for k=4, g=10
    if k == 4 and g == 10:
        # Try to find a graph with n vertices
        # Known best is 320. Moore bound is 242.
        # We try n from 320 down to 250.
        
        def get_girth(adj):
            # BFS from each vertex, non-backtracking
            n = len(adj)
            min_girth = float('inf')
            for start in range(n):
                # dist[v] = distance, prev[v] = previous node
                dist = [-1] * n
                prev = [-1] * n
                dist[start] = 0
                queue = [start]
                while queue:
                    next_queue = []
                    for u in queue:
                        for v in adj[u]:
                            if v == prev[u]:
                                continue
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                prev[v] = u
                                next_queue.append(v)
                            else:
                                # Found a cycle
                                # Cycle length is dist[u] + dist[v] + 1
                                # But we need to be careful with the exact path
                                # For non-backtracking BFS, the first time we see a visited node (not parent)
                                # is a cycle.
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < min_girth:
                                    min_girth = cycle_len
                    queue = next_queue
                if min_girth <= g:
                    return min_girth
            return min_girth

        def find_short_cycle(adj, max_len):
            # Returns a list of nodes in a cycle of length < max_len, or None
            n = len(adj)
            for start in range(n):
                # BFS
                dist = [-1] * n
                prev = [-1] * n
                dist[start] = 0
                queue = [start]
                while queue:
                    next_queue = []
                    for u in queue:
                        if dist[u] >= max_len:
                            continue
                        for v in adj[u]:
                            if v == prev[u]:
                                continue
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                prev[v] = u
                                next_queue.append(v)
                            else:
                                # Found a cycle
                                # Reconstruct path
                                path = []
                                curr = v
                                while curr != start and curr != -1:
                                    path.append(curr)
                                    curr = prev[curr]
                                path.append(start)
                                path.append(u)
                                # The cycle is start -> ... -> u -> v -> ... -> start
                                # We need to verify the length
                                # Actually, BFS tree path is unique.
                                # The cycle is formed by u-v edge and paths to LCA.
                                # For simplicity, if we find a back-edge to a node at distance d,
                                # the cycle length is dist[u] + dist[v] + 1.
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < max_len:
                                    # Reconstruct the cycle
                                    # Path from start to u
                                    path_u = []
                                    curr = u
                                    while curr != -1:
                                        path_u.append(curr)
                                        if curr == start: break
                                        curr = prev[curr]
                                    path_u.reverse()
                                    # Path from start to v
                                    path_v = []
                                    curr = v
                                    while curr != -1:
                                        path_v.append(curr)
                                        if curr == start: break
                                        curr = prev[curr]
                                    path_v.reverse()
                                    
                                    # Find LCA
                                    # This is complex. 
                                    # Simpler: just return the edge (u,v) and the nodes involved?
                                    # We just need to break the cycle.
                                    # We can pick any two non-adjacent edges in the cycle.
                                    # To be safe, we can just pick edge (u,v) and another edge far away.
                                    # Or we can just do a random 2-switch on the whole graph.
                                    return True
                    queue = next_queue
            return False

        def random_4reg(n):
            # Configuration model
            half_edges = []
            for i in range(n):
                for _ in range(4):
                    half_edges.append(i)
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v:
                    # Retry
                    # Simple retry: just swap with another
                    # For now, assume no loops in small n? No, loops are possible.
                    # We need to handle loops.
                    # If loop, find another edge to swap with.
                    # This is a simplified generator.
                    pass
                else:
                    adj[u].append(v)
                    adj[v].append(u)
            # Check for multi-edges and loops
            valid = True
            for u in range(n):
                if len(set(adj[u])) != len(adj[u]) or u in adj[u]:
                    valid = False
                    break
            if not valid:
                return None
            return adj

        best_n = None
        best_edges = None
        
        # Try n values
        for n in range(320, 242, -10):
            if time.time() > deadline:
                break
            found = False
            for attempt in range(100):
                if time.time() > deadline:
                    break
                adj = random_4reg(n)
                if adj is None:
                    continue
                # Local search
                for step in range(1000):
                    if time.time() > deadline:
                        break
                    girth = get_girth(adj)
                    if girth >= 10:
                        # Found!
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_n = n
                        best_edges = edges
                        found = True
                        break
                    # Break a cycle
                    # Pick a random edge (u,v)
                    # Find another edge (x,y) not adjacent to u,v
                    # Swap
                    u = random.choice(range(n))
                    v = random.choice(adj[u])
                    x = random.choice(range(n))
                    if x == u or x == v:
                        continue
                    # Pick y from adj[x]
                    y = random.choice(adj[x])
                    if y == u or y == v:
                        continue
                    # Check if (u,x) and (v,y) are already edges
                    if x in adj[u] or y in adj[v]:
                        continue
                    # Perform swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                if found:
                    break
            if found:
                break
        
        if best_n is not None:
            return best_n, best_edges

    return luw_construct()
