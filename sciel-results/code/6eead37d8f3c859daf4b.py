import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Local search for smaller n
    start_time = time.time()
    deadline = start_time + 110  # 40s margin for safety

    def generate_random_regular_graph(n, k):
        # Simple rejection sampling
        while True:
            adj = [[] for _ in range(n)]
            edges = []
            remaining = [k] * n
            # Shuffle vertices
            verts = list(range(n))
            random.shuffle(verts)
            success = True
            for u in verts:
                if remaining[u] == 0:
                    continue
                # Find k - remaining[u] neighbors
                # Actually, we need to add edges.
                # This simple method is prone to failure for high k/n.
                # Let's use a better method: Havel-Hakimi style or just try to connect.
                pass
            
            # Better: Generate random permutation and pair up stubs.
            # Stub matching
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            edges = set()
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    success = False
                    break
                if v in adj[u]:
                    success = False
                    break
                adj[u].append(v)
                adj[v].append(u)
                edges.add((min(u,v), max(u,v)))
            if success:
                return adj, list(edges)
                
    def get_girth(adj, g_max):
        n = len(adj)
        min_girth = float('inf')
        # BFS from each node up to depth g_max // 2
        half_g = g_max // 2
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            parent[start] = start
            queue = deque([start])
            found_cycle = False
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # We need to find the LCA to get exact length, 
                        # but for "shortest cycle" check, 
                        # if we find any back edge, the cycle length is 
                        # dist[u] + dist[v] + 1 - 2*dist[lca]
                        # Since we are doing BFS, dist[v] <= dist[u] + 1
                        # Simple check: if we just want to know if girth < g_max
                        cycle_len = dist[u] + dist[v] + 1
                        # This is an upper bound on the cycle length via this edge.
                        # Actually, the cycle formed is u-v-parent...-u
                        # The length is dist[u] + dist[v] + 1 - 2*dist[lca]
                        # To be safe, if dist[u] + dist[v] + 1 < g_max, we have a short cycle?
                        # No, dist[u]+dist[v]+1 is the length of the path u->start->v + edge (u,v).
                        # This forms a cycle of length dist[u] + dist[v] + 1.
                        # This cycle might not be the shortest, but it is A cycle.
                        # If this length < g_max, then girth < g_max.
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            if min_girth < g_max:
                break
        return min_girth

    def find_graph(n, k, g, time_limit):
        start_t = time.time()
        # Try a few random instances
        for _ in range(10):
            if time.time() - start_t > time_limit:
                break
            adj, edges = generate_random_regular_graph(n, k)
            girth = get_girth(adj, g)
            if girth >= g:
                return n, edges
            
            # Local search
            while time.time() - start_t < time_limit:
                girth = get_girth(adj, g)
                if girth >= g:
                    return n, edges
                
                # Find a short cycle
                # We need the actual cycle to break it.
                # For now, just do random edge swaps and hope.
                # Pick two random edges (u,v) and (x,y)
                # Remove them, add (u,x) and (v,y) or (u,y) and (v,x)
                # Check validity
                u, v = random.choice(edges)
                x, y = random.choice(edges)
                if u == x or u == y or v == x or v == y:
                    continue
                # Try swap 1
                # Remove (u,v) and (x,y)
                # Add (u,x) and (v,y)
                # Check if (u,x) and (v,y) exist
                if x in adj[u] or y in adj[v] or x in adj[v] or y in adj[u]:
                    continue
                
                # Apply
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(y)
                adj[y].append(v)
                
                # Update edges list
                # This is inefficient. Let's just rebuild or maintain a set.
                # For simplicity in this snippet, we'll just return if we find one.
                # The local search logic here is weak.
                pass

    # The local search above is too weak. 
    # Given the complexity, and the risk of timeout, 
    # and the fact that the LUW graph is valid, 
    # I will stick to the LUW graph but try to optimize it or use a known smaller one if I can recall one.
    # However, 6250 is the current score.
    # If I can't find a better one quickly, I should return the LUW graph.
    
    # Let's try a specific known graph if possible.
    # The (5,9) cage is not well-known in standard libraries.
    
    # I will return the LUW graph to ensure correctness and avoid timeout.
    return luw_construct(k, g)
