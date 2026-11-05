import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback: The Lazebnik-Ustimenko-Woldar construction
    def l_u_w_construct(k, g):
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

    def has_short_cycle(adj, n, g):
        """Check if graph has any cycle shorter than g using BFS from each vertex."""
        for start in range(n):
            # BFS to find shortest cycle through start
            # For each neighbor of start, BFS to find path back to start
            visited = [-1] * n
            parent = [-1] * n
            queue = deque()
            for nb in adj[start]:
                visited[nb] = start
                parent[nb] = start
                queue.append(nb)
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = u
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        # Calculate cycle length
                        a, b = u, v
                        length = 1  # edge u-v
                        while a != b:
                            if visited[a] > visited[b]:
                                a = parent[a]
                                length += 1
                            else:
                                b = parent[b]
                                length += 1
                        if length < g:
                            return True
        return False

    def try_repair(n, k, g, timeout=10):
        """Generate random k-regular graph and repair short cycles."""
        start_time = time.time()
        target_edges = (n * k) // 2
        
        # Generate random k-regular graph using configuration model
        adj = [[] for _ in range(n)]
        half_edges = list(range(target_edges))
        
        # Create stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        # Pair up stubs
        edges_set = set()
        i = 0
        while i < len(stubs):
            u, v = stubs[i], stubs[i + 1]
            i += 2
            if u == v:
                # Self-loop, retry
                # Find another partner for u
                for j in range(i, len(stubs)):
                    if stubs[j] != u:
                        stubs[i], stubs[j] = stubs[j], stubs[i]
                        break
                u, v = stubs[i], stubs[i + 1]
                i += 2
            key = (min(u, v), max(u, v))
            if key in edges_set:
                # Duplicate, retry
                for j in range(i, len(stubs)):
                    if stubs[j] != v:
                        stubs[i], stubs[j] = stubs[j], stubs[i]
                        break
                u, v = stubs[i], stubs[i + 1]
                i += 2
            edges_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Now repair: find and break short cycles
        iterations = 0
        max_iterations = 50000
        while has_short_cycle(adj, n, g) and iterations < max_iterations:
            if time.time() - start_time > timeout:
                return None
            iterations += 1
            
            # Find a short cycle
            cycle = None
            for start in range(n):
                visited = [-1] * n
                parent = [-1] * n
                queue = deque()
                for nb in adj[start]:
                    visited[nb] = start
                    parent[nb] = start
                    queue.append(nb)
                while queue and cycle is None:
                    u = queue.popleft()
                    for v in adj[u]:
                        if visited[v] == -1:
                            visited[v] = u
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            a, b = u, v
                            length = 1
                            path = [a]
                            while a != b:
                                if visited[a] > visited[b]:
                                    a = parent[a]
                                    path.append(a)
                                    length += 1
                                else:
                                    b = parent[b]
                                    length += 1
                            if length < g:
                                cycle = path
                                break
                if cycle is not None:
                    break
            
            if cycle is None:
                break
            
            # Remove a random edge from the cycle and replace it
            # Pick a random edge in the cycle to remove
            edge_idx = random.randint(0, len(cycle) - 1)
            u = cycle[edge_idx]
            v = cycle[(edge_idx + 1) % len(cycle)]
            
            # Remove edge (u, v)
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Find a new edge to add: pick two vertices with degree < k
            candidates_u = [i for i in range(n) if len(adj[i]) < k]
            candidates_v = [i for i in range(n) if len(adj[i]) < k]
            
            if not candidates_u or not candidates_v:
                # Restore edge
                adj[u].append(v)
                adj[v].append(u)
                return None
            
            # Try random pairs
            found = False
            for _ in range(100):
                ru = random.choice(candidates_u)
                rv = random.choice(candidates_v)
                if ru == rv:
                    continue
                if rv in adj[ru]:
                    continue
                # Check if adding this edge creates a short cycle
                # Quick check: BFS from ru to rv
                max_depth = g - 2
                visited = {ru: 0}
                q = deque([(ru, 0)])
                valid = True
                while q:
                    curr, depth = q.popleft()
                    if depth >= max_depth:
                        continue
                    for nei in adj[curr]:
                        if nei == rv:
                            valid = False
                            break
                        if nei not in visited:
                            visited[nei] = depth + 1
                            q.append((nei, depth + 1))
                    if not valid:
                        break
                if valid:
                    adj[ru].append(rv)
                    adj[rv].append(ru)
                    found = True
                    break
            
            if not found:
                # Restore edge
                adj[u].append(v)
                adj[v].append(u)
                return None
        
        # Final check
        if has_short_cycle(adj, n, g):
            return None
        
        # Build edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        if len(edges) != target_edges:
            return None
        
        return n, edges

    if k == 5 and g == 9:
        # Try repair approach with various n
        for n in [1000, 1050, 1100, 1116]:
            res = try_repair(n, k, g, timeout=15)
            if res:
                return res
    else:
        # Generic case
        moore_bound = 0
        if g % 2 == 1:
            r = (g - 3) // 2
            for i in range(r):
                moore_bound += (k - 1) ** i
            moore_bound *= k
        else:
            r = g // 2 - 1
            for i in range(r + 1):
                moore_bound += (k - 1) ** i
            moore_bound *= 2
        
        n_start = max(10, moore_bound)
        for n in [n_start, n_start + 50, n_start + 100]:
            res = try_repair(n, k, g, timeout=10)
            if res:
                return res

    return l_u_w_construct(k, g)
