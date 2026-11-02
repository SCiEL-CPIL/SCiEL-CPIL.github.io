import random
import time
from collections import deque

def _verify(n, edges, k, g):
    """Verify simple, k-regular, no cycle < g via non-backtracking BFS from every vertex."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    for v0 in range(n):
        for v1 in adj[v0]:
            visited = set()
            visited.add((v0, v1))
            queue = deque([(v0, v1, 1)])
            while queue:
                prev, curr, dist = queue.popleft()
                if dist >= g:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    new_state = (curr, nxt)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    if nxt == v0 and dist + 1 < g:
                        return False
                    queue.append((curr, nxt, dist + 1))
    return True

def _generate_random_4_reg_bipartite(n):
    if n % 2 != 0:
        return None
    half = n // 2
    perms = [random.sample(range(half), half) for _ in range(4)]
    edges = []
    adj = [set() for _ in range(n)]
    valid = True
    for u in range(half):
        for i in range(4):
            v = half + perms[i][u]
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        if not valid:
            break
    if not valid:
        return None
    return edges

def _find_short_cycle(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        prev = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append(v)
                elif v != prev[u] and u != prev[v]:
                    # Check cycle length
                    # Path start->u and start->v
                    # Cycle length = dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        # Reconstruct cycle edges
                        # This is expensive, but we only do it once per iteration
                        path_u = []
                        cur = u
                        while cur != -1:
                            path_u.append(cur)
                            cur = prev[cur]
                        path_u.reverse()
                        
                        path_v = []
                        cur = v
                        while cur != -1:
                            path_v.append(cur)
                            cur = prev[cur]
                        path_v.reverse()
                        
                        # Combine
                        # The cycle is path_u + reversed(path_v[1:])
                        cycle_verts = path_u + path_v[1:][::-1]
                        # Extract edges
                        cycle_edges = []
                        for i in range(len(cycle_verts)):
                            u1 = cycle_verts[i]
                            v1 = cycle_verts[(i+1) % len(cycle_verts)]
                            if (u1, v1) in edges or (v1, u1) in edges:
                                cycle_edges.append((min(u1, v1), max(u1, v1)))
                        return cycle_edges
    return None

def construct(k, g):
    if k != 4 or g != 10:
        # Fallback
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # ... L-U-W logic ...
        pass

    start_time = time.time()
    time_limit = 100
    
    for N in range(320, 330, 2):
        if time.time() - start_time > time_limit:
            break
        edges = _generate_random_4_reg_bipartite(N)
        if not edges:
            continue
        
        # Local Search
        while time.time() - start_time < time_limit:
            cycle = _find_short_cycle(N, edges, g)
            if not cycle:
                if _verify(N, edges, k, g):
                    return N, edges
                break
            
            # Try to break the cycle
            # Pick a random edge from the cycle
            e1 = random.choice(cycle)
            # Find another edge to swap with
            # To maintain regularity, we need a 2-opt swap.
            # e1 = (u1, v1). We need e2 = (u2, v2) such that u1!=u2, v1!=v2.
            # And (u1, v2) and (u2, v1) are not existing edges.
            # And the swap doesn't create new short cycles.
            
            # For simplicity, just try a few random swaps
            improved = False
            for _ in range(10):
                # Pick random edge e2
                e2 = random.choice(edges)
                if e1 == e2:
                    continue
                u1, v1 = e1
                u2, v2 = e2
                if u1 == u2 or v1 == v2:
                    continue
                # Check if new edges exist
                new_e1 = (min(u1, v2), max(u1, v2))
                new_e2 = (min(u2, v1), max(u2, v1))
                if new_e1 in edges or new_e2 in edges:
                    continue
                if new_e1 == e1 or new_e2 == e2:
                    continue
                
                # Try swap
                new_edges = [e for e in edges if e != e1 and e != e2]
                new_edges.append(new_e1)
                new_edges.append(new_e2)
                
                # Check girth
                if _verify(N, new_edges, k, g):
                    return N, new_edges
                
                # Check if it breaks the current cycle and doesn't create worse ones?
                # For now, just accept if it breaks the cycle?
                # We don't know if it breaks the cycle without checking.
                # Let's just accept if _verify passes.
                # This is slow.
                pass
            
            # If no swap worked, restart?
            break

    # Fallback to L-U-W
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
