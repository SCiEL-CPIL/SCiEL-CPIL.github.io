import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # For 4-regular graph with girth g, use circulant graph on n vertices
    # with jumps ±a, ±b
    # Moore bound for even girth g=2m: n >= 2 * sum_{i=0}^{m-1} (k-1)^i
    m = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(m))
    
    # Try circulant graphs: vertices 0..n-1, edges (v, v±a), (v, v±b)
    # Need: no cycle of length < g
    
    def check_girth(n, a, b, required_girth):
        """Check if the 4-regular circulant graph with jumps ±a, ±b on n vertices
        has girth >= required_girth. Uses BFS from each vertex to find shortest cycle."""
        # For efficiency, we only need to check from a subset of vertices
        # if the graph is vertex-transitive (which circulants are)
        # So we just check from vertex 0
        # But we need to be careful: the girth is the shortest cycle anywhere.
        # For a vertex-transitive graph, checking from one vertex suffices for
        # the girth? No, that's not quite right. Let me just do BFS from each vertex
        # but with early termination.
        
        # Actually for a circulant graph, it's vertex-transitive, so the girth
        # can be found by checking from vertex 0.
        # BFS from 0, tracking parent, and looking for back edges.
        
        visited = [-1] * n  # visited[v] = distance from 0
        parent = [-1] * n
        parent_edge = [-1] * n  # which edge index was used
        visited[0] = 0
        queue = deque([0])
        
        # The jumps
        jumps = [a, -a, b, -b]
        
        while queue:
            u = queue.popleft()
            for j, jump in enumerate(jumps):
                v = (u + jump) % n
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    parent_edge[v] = j
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    if u != v and parent[v] != u and parent[u] != v:
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        cycle_len = visited[u] + visited[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True
    
    # Search for valid (n, a, b)
    best_result = None
    
    # Try n values starting from Moore bound
    # For efficiency, try a range of n values
    max_n = moore_bound + 200  # try up to this many extra vertices
    
    # For each n, try random pairs (a, b)
    # a should be coprime to n, b should be coprime to n, a != b, a != -b mod n
    
    for n in range(moore_bound, max_n + 1):
        if time.time() > deadline:
            break
        
        # Try a limited number of random pairs
        tried = 0
        max_tries = 50  # per n value
        
        while tried < max_tries:
            if time.time() > deadline:
                break
            tried += 1
            
            a = random.randint(1, n - 1)
            b = random.randint(1, n - 1)
            
            # Ensure a != b, a != -b mod n, and both coprime to n
            if a == b or a == (n - b) % n:
                continue
            if a == 0 or b == 0:
                continue
            
            # Check gcd
            import math
            if math.gcd(a, n) != 1 or math.gcd(b, n) != 1:
                continue
            
            if check_girth(n, a, b, g):
                # Build the edge list
                edges = []
                for v in range(n):
                    for jump in [a, -a, b, -b]:
                        u = (v + jump) % n
                        if v < u:  # avoid duplicate edges
                            edges.append((v, u))
                
                # Verify it's 4-regular and simple
                # Each vertex should have exactly 4 neighbors
                # Since we have ±a and ±b, and a != ±b, each vertex has 4 distinct neighbors
                # unless a = n/2 or b = n/2 (which would make a = -a mod n)
                if n % 2 == 0 and (a == n // 2 or b == n // 2):
                    continue
                
                # Verify no duplicate edges
                edge_set = set()
                valid = True
                for e in edges:
                    if e in edge_set or (e[1], e[0]) in edge_set:
                        valid = False
                        break
                    edge_set.add(e)
                if not valid:
                    continue
                
                # Verify 4-regular
                degree = [0] * n
                for u, v in edges:
                    degree[u] += 1
                    degree[v] += 1
                if any(d != k for d in degree):
                    continue
                
                best_result = (n, edges)
                break  # Found valid graph for this n
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: use the original LUW construction
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
    
    return best_result
