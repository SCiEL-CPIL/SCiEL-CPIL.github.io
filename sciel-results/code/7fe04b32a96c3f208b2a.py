"""Combined program: L-U-W fallback + random regular graph search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    def check_girth(n, adj, min_girth):
        """Check if graph has girth >= min_girth. Returns True if valid."""
        for s in range(n):
            if time_left() < 5.0:
                return None  # timeout
            visited = [-1] * n
            parent = [-1] * n
            visited[s] = 0
            parent[s] = -1
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if visited[u] >= (min_girth - 1) // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = visited[u] + visited[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    def generate_random_regular_graph(n, k, seed=None):
        """Generate a random k-regular graph using configuration model with repair."""
        if seed is not None:
            random.seed(seed)
        
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        # Try multiple random pairings
        for attempt in range(50):
            if time_left() < 10.0:
                return None
            
            random.shuffle(stubs)
            edges = []
            valid = True
            
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                edges.append((u, v))
            
            if not valid:
                continue
            
            # Check for multi-edges
            edge_set = set()
            valid = True
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
            
            if not valid:
                continue
            
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check degrees
            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
            
            if valid:
                return n, edges, adj
        
        return None
    
    # Strategy: Try random regular graphs starting from near Moore bound
    # If we find a valid one, return it. Otherwise, fall back to L-U-W.
    
    # First, try a few random graphs at various sizes
    sizes_to_try = []
    # Start from Moore bound and go up in steps
    for delta in [0, 10, 20, 50, 100, 200]:
        n_try = moore_bound + delta
        if n_try * k % 2 != 0:  # Must have even number of stubs
            n_try += 1
        sizes_to_try.append(n_try)
    
    # Deduplicate and sort
    sizes_to_try = sorted(set(sizes_to_try))
    
    # Try random graphs for each size
    for n_try in sizes_to_try:
        if time_left() < 15.0:
            break
        if n_try > 2000:  # Don't go too large
            break
        
        result = generate_random_regular_graph(n_try, k)
        if result is None:
            continue
        
        n, edges, adj = result
        
        # Check girth
        girth_valid = check_girth(n, adj, g)
        if girth_valid is True:
            # Found a valid graph!
            # Ensure edges are only listed once (from lower to higher index)
            unique_edges = []
            edge_set = set()
            for u, v in edges:
                if u < v:
                    key = (u, v)
                else:
                    key = (v, u)
                if key not in edge_set:
                    edge_set.add(key)
                    unique_edges.append(key)
            return n, unique_edges
    
    # Fallback: L-U-W construction
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
    
    i = 0
    while i < len(order):
        if time_left() < 2.0:
            break
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
