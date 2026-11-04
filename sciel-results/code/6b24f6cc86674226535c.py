import time
import random
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before the 120s limit

    def check_girth(n, edges, min_girth):
        """Check if the graph has girth at least min_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    elif dist[v] > dist[u] + 1:
                        return False
                    elif dist[v] == dist[u] + 1:
                        # Check for cycle
                        if dist[u] + 1 < min_girth:
                            return False
        return True

    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph construction."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
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
        
        n = len(order)
        return n, edges

    # For k=4, g=10, try to find a smaller graph
    if k == 4 and g == 10:
        # First, get the LUW construction as a baseline
        n_luw, edges_luw = luw_construct(k, g)
        best_n, best_edges = n_luw, edges_luw
        
        # Try to find a smaller graph using a different approach
        # The (4,10)-cage is known to have 320 vertices, but we may not be able to construct it easily.
        # Try a few random 4-regular graphs and check their girth.
        
        def try_random_graph(n_vertices, max_attempts=50):
            """Try to construct a random 4-regular graph with girth >= g."""
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                # Generate a random 4-regular graph
                edges = generate_random_4_regular(n_vertices)
                if edges is None:
                    continue
                # Check girth
                if check_girth(n_vertices, edges, g):
                    return n_vertices, edges
            return None, None
        
        def generate_random_4_regular(n):
            """Generate a random 4-regular graph on n vertices."""
            if n < 4 or n % 2 != 0:
                return None
            
            # Start with a cycle
            edges = set()
            for i in range(n):
                edges.add((i, (i + 1) % n))
            
            # Add 3 more edges per vertex randomly
            # This is a simplified approach; a proper 4-regular graph generation is more complex
            # For now, just return the cycle (which is 2-regular, not 4-regular)
            # This won't work, so let's use a different approach.
            
            # Use the configuration model
            # Each vertex has 4 stubs
            stubs = []
            for i in range(n):
                for _ in range(4):
                    stubs.append(i)
            
            random.shuffle(stubs)
            
            edges = set()
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    return None  # Loop, invalid
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    return None  # Multiple edge, invalid
                edges.add(edge)
            
            # Check degree
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != 4 for d in deg):
                return None
            
            return list(edges)
        
        # Try a few sizes
        for n_try in range(243, min(n_luw, 350)):
            if time.time() > deadline:
                break
            if n_try % 2 != 0:
                continue  # 4-regular graph needs even number of vertices
            result_n, result_edges = try_random_graph(n_try, max_attempts=10)
            if result_n is not None:
                best_n, best_edges = result_n, result_edges
                break
        
        return best_n, best_edges
    else:
        return luw_construct(k, g)
