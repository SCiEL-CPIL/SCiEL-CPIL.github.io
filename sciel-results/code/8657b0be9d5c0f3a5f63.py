import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Special cases for small k, g
    special_cases = {
        (3, 5): (10, Petersen_edges()),
        (3, 6): (14, Heawood_edges()),
    }
    
    if (k, g) in special_cases:
        return special_cases[(k, g)]

    # For k=4, g=10, try to find a 320-vertex graph
    if k == 4 and g == 10:
        # Attempt Local Search for N=320
        # We have ~110s budget
        search_start = time.time()
        best_n = None
        best_edges = None
        
        # Try multiple random restarts
        while time.time() < deadline - 5.0:
            n, edges = _local_search_cage(4, 10, 320)
            if edges is not None:
                # Verify quickly
                if _fast_verify_girth(n, edges, 4, 10):
                    return n, edges
        
        # Fallback to LUW
        luw_n, luw_edges = _luw_construct(4, 10)
        return luw_n, luw_edges

    # General case: LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
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

def _random_k_regular(n, k):
    """Generate a random k-regular simple graph using configuration model with rejection."""
    for attempt in range(100):
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        edges = []
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        
        if valid and all(len(adj[i]) == k for i in range(n)):
            return edges
    
    return None

def _local_search_cage(k, g, n):
    """Try to find a k-regular graph on n vertices with girth >= g."""
    # Generate random k-regular graph
    edges = _random_k_regular(n, k)
    if edges is None:
        return n, None
    
    # Quick girth check
    if _fast_verify_girth(n, edges, k, g):
        return n, edges
    
    return n, None

def _fast_verify_girth(n, edges, k, g):
    """Fast girth verification using non-backtracking walks."""
    # Build adj list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check simple
    edge_set = set()
    for u, v in edges:
        if u > v: u, v = v, u
        if (u, v) in edge_set: return False
        edge_set.add((u, v))
    
    # Use non-backtracking BFS to detect short cycles
    # For each vertex, do BFS with state (current, prev) to avoid backtracking
    max_depth = (g - 1) // 2  # We need to detect cycles of length < g
    
    for start in range(n):
        # State: (current_node, previous_node) -> visited
        visited = set()
        queue = deque()
        
        # Initialize: from start, go to each neighbor
        for nb in adj[start]:
            state = (nb, start)
            visited.add(state)
            queue.append((nb, start, 1))  # (current, prev, depth)
        
        while queue:
            curr, prev, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue  # Non-backtracking
                
                new_state = (nb, curr)
                
                if new_state in visited:
                    # Found a cycle
                    # Cycle length is depth + 1 (edge from curr to nb)
                    # But we need to be careful about the actual cycle length
                    cycle_len = depth + 1
                    if cycle_len < g:
                        return False
                    continue
                
                visited.add(new_state)
                queue.append((nb, curr, depth + 1))
    
    return True

def Petersen_edges():
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    for i in range(5):
        edges.append((i, 5 + i))
    return edges

def Heawood_edges():
    edges = []
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(lines):
        for p in line:
            edges.append((p, 7 + i))
    return edges
