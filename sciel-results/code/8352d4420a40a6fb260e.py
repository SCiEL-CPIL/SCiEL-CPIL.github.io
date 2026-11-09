import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
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

    def random_greedy(n, k, g):
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        adjacency = [[] for _ in range(n)]
        edges_set = set()
        
        # stubs_count[v] = number of stubs left for vertex v
        stubs_count = [k] * n
        active_vertices = list(range(n))
        
        attempts = 0
        max_attempts = n * k * 10 # Heuristic limit
        
        while any(stubs_count):
            if attempts > max_attempts:
                return None
            attempts += 1
            
            # Pick u
            idx_u = random.randint(0, len(active_vertices) - 1)
            u = active_vertices[idx_u]
            
            # Pick v != u
            if len(active_vertices) == 1:
                return None
            idx_v = random.randint(0, len(active_vertices) - 1)
            if idx_v == idx_u:
                idx_v = (idx_v + 1) % len(active_vertices)
            v = active_vertices[idx_v]
            
            # Check if edge exists
            if (u, v) in edges_set or (v, u) in edges_set:
                continue
                
            # Check girth
            if find(u) == find(v):
                # BFS
                visited = {u: 0}
                queue = deque([u])
                dist = -1
                while queue:
                    curr = queue.popleft()
                    d_curr = visited[curr]
                    if curr == v:
                        dist = d_curr
                        break
                    if d_curr >= g - 2:
                        break
                    for neighbor in adjacency[curr]:
                        if neighbor not in visited:
                            visited[neighbor] = d_curr + 1
                            queue.append(neighbor)
                
                if dist != -1 and dist < g - 1:
                    continue
            
            # Accept
            edges_set.add((u, v))
            adjacency[u].append(v)
            adjacency[v].append(u)
            union(u, v)
            
            stubs_count[u] -= 1
            stubs_count[v] -= 1
            
            if stubs_count[u] == 0:
                active_vertices.pop(idx_u)
                if idx_v > idx_u:
                    idx_v -= 1
            if stubs_count[v] == 0:
                active_vertices.pop(idx_v)
                
        return adjacency, list(edges_set)

    start_time = time.time()
    deadline = start_time + 110
    
    if g % 2 == 0:
        r = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    else:
        r = (g - 1) // 2
        moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))
        
    moore_bound = max(moore_bound, 2)
    
    current_n = moore_bound
    max_n_to_try = 2000
    
    while current_n <= max_n_to_try and time.time() < deadline:
        result = random_greedy(current_n, k, g)
        if result:
            adjacency, edges = result
            return current_n, edges
        current_n += 1
        
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
