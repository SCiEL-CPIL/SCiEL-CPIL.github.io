import time
import random
import heapq

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _greedy_construct(k, g, n, time_limit=10.0):
    # Try to construct a k-regular graph of girth >= g on n vertices
    start_time = time.time()
    adjacency = [set() for _ in range(n)]
    degrees = [0] * n
    
    # Randomize the order of vertices to break symmetry
    vertices = list(range(n))
    random.shuffle(vertices)
    
    # We need to add k * n / 2 edges
    # We process vertices, adding edges to other vertices
    # To ensure regularity, we must be careful.
    # A simple greedy might get stuck with some vertices having < k and others > k (impossible)
    # or some vertices unable to reach k.
    
    # Better approach: 
    # 1. Generate a random k-regular graph (e.g. using configuration model + 2-switches to make simple)
    # 2. Check girth.
    # 3. If girth < g, try to fix by switching.
    # This is complex.
    
    # Alternative: 
    # Start with empty graph.
    # Maintain a list of "active" vertices (degree < k).
    # While active:
    #   Pick u from active.
    #   Pick v from active, v != u.
    #   If (u,v) is not an edge and adding it doesn't create short cycle:
    #      Add edge.
    #      Update degrees.
    #      If degree[u] == k, remove from active.
    #      If degree[v] == k, remove from active.
    #   Else:
    #      Try next v.
    #      If no v works, restart or fail.
    
    # This greedy might fail to complete the graph (stuck with active vertices that cannot connect).
    # To mitigate, we can allow "retries" within the time limit.
    
    def has_short_cycle(u, v, max_depth):
        # BFS from u to see if v is reached within max_depth
        # If yes, adding (u,v) creates a cycle of length <= max_depth + 1
        # We need cycle >= g, so path length must be >= g - 1
        # So we reject if dist(u,v) < g - 1
        if u == v: return True
        visited = {u: 0}
        queue = [(u, 0)]
        while queue:
            node, dist = queue.pop(0)
            if dist >= max_depth:
                continue
            for neighbor in adjacency[node]:
                if neighbor == v:
                    return True # Found v, so path exists
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return False

    active = list(range(n))
    random.shuffle(active)
    
    # We need to be careful with the "active" list management
    # Using a set for active vertices for O(1) removal?
    # But we need to pick randomly. List is fine for n~1000.
    
    while active:
        if time.time() - start_time > time_limit:
            return None
            
        # Pick a vertex u from active
        u_idx = random.randint(0, len(active) - 1)
        u = active[u_idx]
        
        # Find a valid v
        found = False
        # Shuffle active to avoid bias
        active_copy = active[:]
        random.shuffle(active_copy)
        
        for v in active_copy:
            if v == u:
                continue
            if v in adjacency[u]:
                continue
            
            # Check if adding (u,v) creates a short cycle
            # We need to ensure no path of length < g-1 between u and v
            if has_short_cycle(u, v, g - 2):
                continue
            
            # Add edge
            adjacency[u].add(v)
            adjacency[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
            # Update active list
            # Remove u if degree == k
            # Remove v if degree == k
            # This is inefficient if we do it every time.
            # Instead, we can just check degrees when picking.
            # But `active` contains vertices with degree < k.
            # If we add an edge, their degrees increase.
            # If they reach k, they should be removed.
            
            found = True
            break
            
        if not found:
            # Failed to find a partner for u
            # Restart
            return None
            
        # Clean up active list
        # This is O(n). Acceptable for n=1000.
        new_active = []
        for node in active:
            if degrees[node] < k:
                new_active.append(node)
        active = new_active
        
    # Verify final graph
    # Check degrees
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
            
    # Check girth (optional, but good for safety)
    # The greedy check should have ensured it.
    
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append((i, j))
                
    return n, edges

def construct(k, g):
    # For the specific case k=5, g=9, try greedy
    if k == 5 and g == 9:
        # Try a few sizes
        # The best known is 1116.
        # We try slightly larger to increase probability of success.
        target_ns = [1116, 1120, 1150, 1200, 1500]
        
        # Time budget: 100s for greedy, 10s for fallback
        time_per_attempt = 10.0 
        
        for n in target_ns:
            res = _greedy_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
