import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback: LUW construction
    def luw_construct(k, g):
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

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # Voltage graph search on K_3,3
    # Base graph K_3,3
    # U: 0,1,2 ; V: 3,4,5
    # Edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    base_edges = [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)]
    
    def check_girth(n, adj, target_girth):
        # Check for cycles shorter than target_girth
        # BFS from each vertex
        for u in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[u] = 0
            q = deque([(u, 0)])
            while q:
                v, dist = q.popleft()
                if dist >= target_girth // 2 - 1:
                    continue
                for w in adj[v]:
                    if visited[w] == -1:
                        visited[w] = dist + 1
                        parent[w] = v
                        q.append((w, dist + 1))
                    elif w != parent[v] and dist + 1 - visited[w] < target_girth:
                        # Found a cycle of length (dist + 1 - visited[w]) + 2?
                        # Actually, if we see a back-edge to a non-parent, cycle length is dist + 1 - visited[w]
                        # But we need to be careful.
                        # Simpler: if we find any cycle < target_girth, return False
                        # This BFS finds shortest path. If shortest path from u to w is d, and we have edge v-w,
                        # and v is at dist d-1, then cycle length is d+1?
                        # Let's just check if we can reach u with length < target_girth?
                        # No, girth is the shortest cycle.
                        # If we do BFS from u, and we find a cycle, its length is at least the distance in BFS tree + 1.
                        # To be safe, we can just run BFS and if we find a cycle, calculate its length.
                        # But this is slow.
                        pass
        return True

    # Better girth check:
    # Count closed walks of length L using adjacency matrix?
    # For n=6000, matrix is 6000x6000. Too big for dense.
    # Use sparse?
    # Let's use the BFS method but optimize it.
    
    def has_short_cycle(adj, max_len):
        n = len(adj)
        # Check for 4-cycles
        # For each edge (u,v), check if they share a neighbor
        # This is O(E * deg)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Check common neighbors
                    common = set(adj[u]).intersection(adj[v])
                    if common:
                        return True
        # Check for 6-cycles
        # For each edge (u,v), check if there is a path of length 4
        # This is harder.
        # Let's just do BFS up to depth max_len//2
        for u in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[u] = 0
            q = deque([(u, 0)])
            while q:
                v, dist = q.popleft()
                if dist >= max_len // 2:
                    continue
                for w in adj[v]:
                    if visited[w] == -1:
                        visited[w] = dist + 1
                        parent[w] = v
                        q.append((w, dist + 1))
                    elif w != parent[v]:
                        # Cycle found
                        cycle_len = dist + 1 - visited[w]
                        if cycle_len < max_len:
                            return True
        return False

    best_n = float('inf')
    best_edges = None

    # Try voltage graphs
    for m in range(100, 2000, 10):
        if time.time() > deadline:
            break
        n = 6 * m
        if n >= best_n:
            continue
            
        # Try random voltages
        for attempt in range(20):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m-1) for _ in range(9)]
            
            # Build graph
            adj = [[] for _ in range(n)]
            edges = []
            for i, (u, v) in enumerate(base_edges):
                w = voltages[i]
                for alpha in range(m):
                    u_lift = u * m + alpha
                    v_lift = v * m + ((alpha + w) % m)
                    adj[u_lift].append(v_lift)
                    adj[v_lift].append(u_lift)
                    edges.append((u_lift, v_lift))
            
            # Check girth
            if not has_short_cycle(adj, 16):
                best_n = n
                best_edges = edges
                return best_n, best_edges

    if best_edges is not None:
        return best_n, best_edges

    return luw_construct(k, g)
