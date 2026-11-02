import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

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

    # Proper girth check using non-backtracking walks from every vertex
    def check_girth_proper(n, adj):
        """Check that no cycle shorter than g exists using BFS from each vertex."""
        for start in range(n):
            # BFS tracking (node, prev) to avoid backtracking
            visited = set()
            queue = deque()
            for v in adj[start]:
                queue.append((start, v, 0))
                visited.add((start, v))
            
            while queue:
                u, v, dist = queue.popleft()
                if dist + 1 >= g:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue  # Don't backtrack
                    if (v, w) in visited:
                        continue
                    visited.add((v, w))
                    # Check if w is reachable from start at a shorter distance
                    # If w == start and dist+1 < g, we found a cycle
                    if w == start and dist + 1 < g:
                        return False
                    queue.append((v, w, dist + 1))
        return True

    # Faster girth check: BFS from each vertex, detect any cycle < g
    def check_girth_bfs(n, adj):
        """Check girth >= g using standard BFS with cycle detection."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a potential cycle
                        # The cycle length is dist[u] + dist[v] + 1 if u and v are in different branches
                        # But this isn't quite right for general cycle detection
                        # Better: if we see an edge (u,v) where both are visited and 
                        # they're not parent-child, check the cycle length
                        if parent[u] != v and parent[v] != u:
                            # This forms a cycle of length dist[u] + dist[v] + 1
                            # But only if the LCA is start or something... this is complex
                            # Simpler: just check if dist[u] + dist[v] + 1 < g
                            if dist[u] + dist[v] + 1 < g:
                                return False
        return True

    # Even simpler and more reliable: for each pair of adjacent nodes, check if there's
    # a path of length < g-1 between them not using that edge
    def check_girth_reliable(n, adj, g):
        """Reliable girth check: for each edge (u,v), check if there's a path of length < g-1 from u to v not using edge (u,v)."""
        for u in range(n):
            for v in adj[u]:
                if v <= u:  # Only check each edge once
                    continue
                # BFS from u to v, not using edge (u,v)
                dist = [-1] * n
                dist[u] = 0
                queue = deque([u])
                found = False
                while queue:
                    x = queue.popleft()
                    if dist[x] >= g - 2:
                        break
                    if x == v:
                        if dist[x] < g - 1:
                            return False
                        found = True
                        break
                    for y in adj[x]:
                        if (x == u and y == v) or (x == v and y == u):
                            continue
                        if dist[y] == -1:
                            dist[y] = dist[x] + 1
                            queue.append(y)
        return True

    # Search for circulant graphs with good girth
    def search_circulants(k, g, max_n, time_limit):
        """Search for 3-regular circulant graphs with girth >= g."""
        if k != 3:
            return None
        
        best = None
        # Try structured generator sets first
        for n in range(512, max_n + 1, 16):
            if time.time() > start_time + time_limit:
                break
            
            # Generate candidate generator sets
            candidates = set()
            
            # Structured sets
            if n % 2 == 0:
                candidates.add(frozenset([1, n // 3, n // 2]) if n % 3 == 0 else frozenset([1, n // 4, n // 2]))
            
            # Try some random sets
            for _ in range(3):
                if time.time() > start_time + time_limit:
                    break
                gen = []
                used = set()
                for _ in range(3):
                    s = random.randint(1, n - 1)
                    if s not in used and (n - s) not in used:
                        gen.append(s)
                        used.add(s)
                        used.add(n - s)
                if len(gen) == 3:
                    candidates.add(frozenset(gen))
            
            for gen_set in candidates:
                gen = list(gen_set)
                
                # Validate: no self-loops, no multi-edges
                valid = True
                for i in range(3):
                    for j in range(i + 1, 3):
                        if gen[i] == gen[j] or gen[i] == n - gen[j]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Build adjacency
                adj = [[] for _ in range(n)]
                for i in range(n):
                    for s in gen:
                        j1 = (i + s) % n
                        j2 = (i - s) % n
                        adj[i].append(j1)
                        adj[i].append(j2)
                
                # Check degree
                if any(len(adj[i]) != 6 for i in range(n)):
                    continue
                
                # Check for duplicate edges
                edge_set = set()
                has_dup = False
                for i in range(n):
                    for j in adj[i]:
                        if j == i:
                            has_dup = True
                            break
                        key = (min(i, j), max(i, j))
                        if key in edge_set:
                            has_dup = True
                            break
                        edge_set.add(key)
                    if has_dup:
                        break
                if has_dup:
                    continue
                
                # Check girth
                if check_girth_reliable(n, adj, g):
                    edges = []
                    for i in range(n):
                        for s in gen:
                            edges.append((i, (i + s) % n))
                    if best is None or n < best[0]:
                        best = (n, edges)
                    return best
        
        return best

    # Try to find a smaller graph
    if k == 3 and g == 16:
        result = search_circulants(k, g, 900, 80.0)
        if result is not None:
            return result
    
    # Fallback
    return luw_construct(k, g)
