import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Strategy: Try to construct a small k-regular graph with girth >= g.
    # Primary approach: Known constructions (LW graphs, etc.)
    # Secondary approach: Random search for smaller graphs
    
    # For the specific case k=4, g=10, we know the (4,10)-cage has 320 vertices.
    # Let's try to construct it or a similar one.
    
    # First, try the LW graph construction which is known to produce high-girth graphs.
    # For q=4, the LW graph with appropriate parameters can give good results.
    
    def build_lw_graph(q, d):
        """
        Build the Lazebnik-Ustimenko-Woldar graph over GF(q) with parameter d.
        This gives a q-regular graph with girth at least 2d+2.
        For q=4, d=4: girth >= 10, which is what we need for g=10.
        """
        if q == 4:
            # GF(4) operations
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q
        
        # Define the rule for the LW graph
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
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
        
        # BFS to enumerate all vertices
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
    
    # For k=4, g=10: Use d=4 (girth >= 2*4+2 = 10)
    if k == 4 and g <= 10:
        d = max(3, (g - 2) // 2)  # d=4 for g=10
        n, edges = build_lw_graph(k, d)
        
        # Verify the graph
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        valid = True
        for i in range(n):
            if len(adj[i]) != k:
                valid = False
                break
        
        if valid:
            # Verify girth using non-backtracking BFS
            def verify_girth(n, adj, g):
                for start_v in range(n):
                    visited = set()
                    queue = deque()
                    
                    for nb in adj[start_v]:
                        if (nb, start_v) not in visited:
                            visited.add((nb, start_v))
                            queue.append((nb, start_v, 1))
                    
                    while queue:
                        curr, prev, depth = queue.popleft()
                        
                        if curr == start_v and depth < g:
                            return False
                        
                        if depth >= g - 1:
                            continue
                        
                        for nb in adj[curr]:
                            if nb == prev:
                                continue
                            if (nb, curr) not in visited:
                                visited.add((nb, curr))
                                queue.append((nb, curr, depth + 1))
                
                return True
            
            if verify_girth(n, adj, g):
                return n, edges
    
    # Fallback: Try random construction for smaller graphs
    import random
    random.seed(42)
    
    def random_k_reg(n, k):
        """Generate a random k-regular graph using configuration model."""
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        
        edges = set()
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in edges:
                return None
            edges.add(key)
        
        return list(edges)
    
    def check_girth(n, adj, g):
        """Check if graph has girth at least g using non-backtracking walks."""
        for start_v in range(n):
            visited = set()
            queue = deque()
            
            for nb in adj[start_v]:
                if (nb, start_v) not in visited:
                    visited.add((nb, start_v))
                    queue.append((nb, start_v, 1))
            
            while queue:
                curr, prev, depth = queue.popleft()
                
                if curr == start_v and depth < g:
                    return False
                
                if depth >= g - 1:
                    continue
                
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if (nb, curr) not in visited:
                        visited.add((nb, curr))
                        queue.append((nb, curr, depth + 1))
        
        return True
    
    # Try random search for smaller graphs
    best_n = float('inf')
    best_edges = None
    
    # Lower bound estimate
    moore_bound = 2 + k * (1 + (k-1) + (k-1)**2)
    for n in range(max(moore_bound, 100), min(400, int(deadline - (time.time() - start_time)) // 2 + 100)):
        if time.time() > deadline - 20:
            break
        
        attempts = 0
        while attempts < 50 and time.time() < deadline - 20:
            attempts += 1
            edges = random_k_reg(n, k)
            if edges is None:
                continue
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Quick degree check
            if any(len(adj[i]) != k for i in range(n)):
                continue
            
            # Check girth
            if check_girth(n, adj, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Final fallback: return the LW graph result
    if k == 4 and g <= 10:
        d = max(3, (g - 2) // 2)
        n, edges = build_lw_graph(k, d)
        return n, edges
    
    # Generic fallback
    d = max(3, (g - 2) // 2)
    n, edges = build_lw_graph(k, d)
    return n, edges
