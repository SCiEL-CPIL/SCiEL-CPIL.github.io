"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # First, generate the base graph using Lazebnik-Ustimenko-Woldar construction
    base_n, base_edges = _generate_luw_graph(k, g)
    
    if time.time() > deadline - 5:
        return base_n, base_edges
    
    best_n = base_n
    best_edges = base_edges
    
    # Try to improve by finding smaller valid graphs through random search
    # Focus on sizes between Moore bound and current best
    moore_bound = _moore_bound(k, g)
    
    # Generate random k-regular graphs and check girth
    attempts = 0
    max_attempts = 30
    
    while time.time() < deadline - 10 and attempts < max_attempts:
        attempts += 1
        
        # Try different sizes starting from Moore bound
        # Focus on sizes that are plausible improvements
        target_n = moore_bound + attempts * 2
        
        if target_n >= best_n:
            continue
            
        # Generate random k-regular graph
        n_try, edges_try = _generate_random_k_regular(target_n, k)
        if n_try is None:
            continue
        
        # Check girth - early exit if we find a short cycle
        girth = _compute_girth_fast(n_try, edges_try, g)
        if girth >= g:
            best_n = n_try
            best_edges = edges_try
            # Reset attempts to try even smaller graphs
            attempts = 0
            max_attempts = 30
    
    return best_n, best_edges

def _generate_luw_graph(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def _moore_bound(k, g):
    # Moore bound for k-regular graphs of girth g
    if g % 2 == 0:
        # Even girth: n >= k + k(k-1)^(g/2 - 1)
        return k + k * (k - 1) ** (g // 2 - 1)
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        total = 1
        term = 1
        for i in range((g - 3) // 2 + 1):
            total += k * term
            term *= (k - 1)
        return total

def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None, None
    
    # Create stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    # Try multiple times to get a simple graph
    for attempt in range(50):
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return n, list(edges)
    
    return None, None

def _compute_girth_fast(n, edges, target_girth):
    """Compute the girth (shortest cycle length) using BFS from each vertex.
       Early exit if girth < target_girth is found."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_cycle = target_girth  # Start with target, we only care if it's smaller
    
    for start in range(n):
        # BFS from start, tracking parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle < target_girth:
                            return min_cycle  # Early exit
    
    return min_cycle
# EVOLVE-BLOCK-END
