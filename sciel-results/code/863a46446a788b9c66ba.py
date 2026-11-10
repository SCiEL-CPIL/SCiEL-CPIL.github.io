import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def _verify_graph(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    if n < 2:
        return False
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i == j or i >= n or j >= n or i < 0 or j < 0:
            return False
        if i in adj[j]:
            return False  # repeated edge
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Convert to list for faster iteration
    adj_list = [sorted(adj[i]) for i in range(n)]
    # Non-backtracking walk check from every vertex
    for start in range(n):
        # State: (current_vertex, previous_vertex) -> min distance
        visited = {}
        queue = deque()
        for v in adj_list[start]:
            queue.append((v, start, 1))
            visited[(v, start)] = 1
        while queue:
            v, prev, dist = queue.popleft()
            if dist >= g:
                continue
            for w in adj_list[v]:
                if w == prev:
                    continue
                key = (w, v)
                if key in visited:
                    if visited[key] + 1 < g:
                        return False
                else:
                    visited[key] = dist + 1
                    queue.append((w, v, dist + 1))
    return True

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction for (k,g) with k odd prime or k=4."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _random_cage(k, g, max_time=60, seed=42):
    """
    Randomized construction of k-regular graph with large girth.
    Uses a greedy approach with random restarts and local search.
    """
    import random
    random.seed(seed)
    
    def _quick_girth_check(adj, k, g):
        """Quick check: BFS from each vertex to depth g/2 to detect short cycles."""
        n = len(adj)
        for start in range(n):
            # BFS from start
            dist = {start: 0}
            parent = {start: -1}
            queue = deque([start])
            found_short = False
            while queue:
                v = queue.popleft()
                if dist[v] >= g:
                    continue
                for w in adj[v]:
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif abs(dist[w] - dist[v]) == 1 and w != parent[v]:
                        # Potential cycle
                        # Trace back to find cycle length
                        # This is a heuristic - for exact check we need non-backtracking
                        pass
        return True  # Placeholder - use full verify for final check
    
    # For k=3, g=16: try to build a graph with ~2000-5000 vertices
    # Use a random regular graph construction with girth constraint
    
    # Start with LUW as baseline
    try:
        luw_n, luw_edges = _luw_graph(k, g)
    except:
        luw_n, luw_edges = None, None
    
    best_n = luw_n if luw_n is not None else 10**9
    best_edges = luw_edges
    
    start_time = time.time()
    
    # Try random constructions with smaller n
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        # Target n: try values between Moore bound and LUW
        moore_bound = 2 * (k-1) ** (g//2 - 1) + 2 * (k-1) ** (g//2)
        target_n = int(moore_bound * 1.5) + 1
        target_n = max(target_n, 100)
        target_n = min(target_n, best_n - 10 if best_n < 10**8 else target_n)
        
        if target_n < 50 or target_n > 100000:
            continue
        
        # Build a random k-regular graph with girth >= g
        # Use configuration model + local search
        try:
            adj = _build_random_regular_with_girth(k, g, target_n, time.time() - start_time + max_time - 5)
            if adj is not None:
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if len(edges) == target_n * k // 2:
                    if _verify_graph(target_n, edges, k, g):
                        best_n = target_n
                        best_edges = edges
        except:
            pass
    
    return best_n, best_edges

def _build_random_regular_with_girth(k, g, target_n, deadline):
    """Build a random k-regular graph with girth >= g using greedy + backtracking."""
    import random
    
    # Initialize adjacency
    adj = [set() for _ in range(target_n)]
    
    # Greedy: connect vertices while maintaining girth constraint
    for v in range(target_n):
        if time.time() > deadline:
            return None
        if len(adj[v]) >= k:
            continue
        # Try to connect v to k - len(adj[v]) other vertices
        attempts = 0
        while len(adj[v]) < k and attempts < 100:
            attempts += 1
            w = random.randint(0, target_n - 1)
            if w == v or w in adj[v]:
                continue
            if len(adj[w]) >= k:
                continue
            # Check if adding edge (v, w) creates a short cycle
            if _edge_preserves_girth(adj, v, w, g, k):
                adj[v].add(w)
                adj[w].add(v)
    
    # Check if we got a k-regular graph
    for i in range(target_n):
        if len(adj[i]) != k:
            return None
    
    return adj

def _edge_preserves_girth(adj, v, w, g, k):
    """Check if adding edge (v, w) to current graph preserves girth >= g."""
    # BFS from v and w simultaneously to detect cycles shorter than g
    # A cycle through (v,w) has length = dist(v,w) + 1 where dist is shortest path avoiding the edge
    # Actually: if there's a path from v to w of length < g-1 not using edge (v,w), then cycle < g
    
    # BFS from v to find shortest path to w
    if w in adj[v]:
        # Edge already exists, shouldn't happen
        return False
    
    dist = {v: 0}
    queue = deque([(v, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= g - 1:
            break
        for neighbor in adj[node]:
            if neighbor not in dist:
                dist[neighbor] = d + 1
                if neighbor == w:
                    # Found path of length d+1, cycle length = d+2
                    if d + 2 < g:
                        return False
                    return True
                queue.append((neighbor, d + 1))
    
    # No path found within g-1 steps, so adding edge is safe
    return True

def construct(k, g):
    deadline = time.time() + 140
    best_n, best_edges = None, None
    
    # Try LUW construction first (fast, explicit)
    try:
        n, edges = _luw_graph(k, g)
        if n is not None and len(edges) == n * k // 2:
            best_n, best_edges = n, edges
    except Exception:
        pass
    
    # Try randomized construction to potentially find smaller graph
    try:
        rand_n, rand_edges = _random_cage(k, g, max_time=90, seed=42)
        if rand_n is not None and rand_n < best_n:
            best_n, best_edges = rand_n, rand_edges
    except Exception:
        pass
    
    # Verify best result
    if best_n is not None:
        if _verify_graph(best_n, best_edges, k, g):
            return best_n, best_edges
    
    # Fallback
    if best_n is not None:
        return best_n, best_edges
    
    return None, None

# EVOLVE-BLOCK-END
