import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def luw_construct(k, g):
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

    def find_shortest_cycle_and_path(adj, n, limit):
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                if dist[u] >= limit:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < limit:
                            # Reconstruct cycle
                            cycle = []
                            curr = u
                            while curr != i:
                                cycle.append(curr)
                                curr = parent[curr]
                            cycle.append(i)
                            
                            curr = v
                            while curr != i:
                                cycle.append(curr)
                                curr = parent[curr]
                            cycle.append(i)
                            # The cycle is u -> ... -> i -> ... -> v -> u
                            # We need the actual path
                            path1 = []
                            curr = u
                            while curr != i:
                                path1.append(curr)
                                curr = parent[curr]
                            path1.append(i)
                            path1.reverse()
                            
                            path2 = []
                            curr = v
                            while curr != i:
                                path2.append(curr)
                                curr = parent[curr]
                            path2.append(i)
                            path2.reverse()
                            
                            full_cycle = path1 + path2[1:]
                            return cycle_len, full_cycle
        return limit, None

    def local_search(N, max_iters):
        # Generate random 3-regular graph
        while True:
            half_edges = []
            for i in range(N):
                half_edges.extend([i] * k)
            random.shuffle(half_edges)
            adj = [[] for _ in range(N)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                break
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            girth, cycle = find_shortest_cycle_and_path(adj, N, g)
            if girth >= g:
                edges = []
                for i in range(N):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return N, edges
            
            if cycle:
                # Break the cycle
                # Pick two non-adjacent edges in the cycle
                # Cycle is a list of vertices
                m = len(cycle)
                if m >= 4:
                    # Pick edge (cycle[0], cycle[1]) and (cycle[m//2], cycle[m//2+1])
                    u1, v1 = cycle[0], cycle[1]
                    u2, v2 = cycle[m//2], cycle[(m//2+1) % m]
                    
                    # Check if swap is valid
                    # Remove (u1, v1) and (u2, v2)
                    # Add (u1, u2) and (v1, v2)
                    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                        continue
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                         # Might be multi-edge
                         pass
                    # Perform swap
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    adj[u1].append(u2)
                    adj[u2].append(u1)
                    adj[v1].append(v2)
                    adj[v2].append(v1)

        return None

    best_result = None
    for N in [936, 1000, 1200]:
        if time.time() > deadline:
            break
        res = local_search(N, 100)
        if res:
            best_result = res
            break
    
    if best_result:
        return best_result
    else:
        return luw_construct(k, g)
