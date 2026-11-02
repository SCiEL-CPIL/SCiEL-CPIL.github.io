import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def has_girth_at_least(adj, n, g):
        """Check if graph has girth >= g using non-backtracking walks from every vertex."""
        # For each vertex, do BFS in the non-backtracking state space
        # State: (current_vertex, previous_vertex) where previous_vertex is the vertex we came from
        # We explore non-backtracking walks and check if we can return to start with walk length < g
        
        half_g = g // 2  # We only need to check up to this depth
        
        for u in range(n):
            # State: (vertex, prev) - we're at vertex, came from prev
            visited = {}  # (vertex, prev) -> distance
            queue = deque()
            
            # Initialize: from u, we can go to any neighbor (no prev constraint for first step)
            for w in adj[u]:
                state = (w, u)
                visited[state] = 1
                queue.append((w, u, 1))
            
            while queue:
                v, prev, dist = queue.popleft()
                
                if dist >= half_g:
                    continue
                
                for w in adj[v]:
                    if w == prev:
                        continue  # non-backtracking
                    if w == u:
                        # Found a cycle of length dist + 1
                        if dist + 1 < g:
                            return False
                        continue
                    state = (w, v)
                    if state in visited:
                        continue
                    visited[state] = dist + 1
                    if dist + 1 < half_g:
                        queue.append((w, v, dist + 1))
        
        return True

    def try_random_3reg(N, k, g, max_attempts, time_limit):
        """Try to find a k-regular graph on N vertices with girth >= g using random configuration model."""
        deadline = time.time() + time_limit
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random k-regular graph using configuration model
            adj = [[] for _ in range(N)]
            edge_set = set()
            half_edges = [i for i in range(N) for _ in range(k)]
            random.shuffle(half_edges)
            
            valid = True
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                continue
            
            # Check girth
            if has_girth_at_least(adj, N, g):
                edges = list(edge_set)
                return N, edges
        
        return None

    # For the specific case k=3, g=16, try to find a smaller graph
    if k == 3 and g == 16:
        # Moore bound: 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254
        # But actual lower bound is 512, smallest known is 936
        # Try various N values with limited time
        remaining = deadline - time.time()
        
        # Try N values from 512 upward, with decreasing time per attempt
        for N in range(512, 1000, 8):
            if time.time() > deadline - 5:  # Keep 5s buffer
                break
            
            # Allocate time based on how much is left
            time_per_N = min(8, max(0.5, (deadline - time.time() - 5) / max(1, (1000 - N) / 8)))
            
            result = try_random_3reg(N, 3, 16, max_attempts=30, time_limit=time_per_N)
            if result is not None:
                return result
    
    # Fallback to LUW construction
    return luw_construct(k, g)
