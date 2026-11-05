import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin under the 150s limit

    # 1. Construct LUW graph as baseline
    # For k=5, g=9, this yields n=1250.
    # This is the best known easy-to-construct graph for this parameter.
    
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        mult_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        def add(x, y): return add_table[x][y]
        def sub(x, y): return add_table[x][y]
        def mul(x, y): return mult_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = [None] * (d + 1)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0: rule[c] = (1, c0 - 2)
        elif r == 1: rule[c] = (c0 - 1, 1)
        elif r == 2: rule[c] = (c0, 1)
        else: rule[c] = (1, c0 + 1)
    
    def get_neighbours(side, x):
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    luw_edges = []
    
    for vertex in order:
        if time.time() > deadline: break
        side, x = vertex
        for other in get_neighbours(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                luw_edges.append((index[vertex], index[other]))
                
    luw_n = len(order)
    
    # Verify the graph is valid (simple, regular, girth >= g)
    adj = [[] for _ in range(luw_n)]
    edges_set = set()
    for u, v in luw_edges:
        if u == v:
            return 0, [] # Invalid
        if (u, v) in edges_set or (v, u) in edges_set:
            return 0, [] # Invalid
        edges_set.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    # Check regularity
    for i in range(luw_n):
        if len(adj[i]) != k:
            return 0, [] # Invalid

    # Check girth using non-backtracking BFS
    # This is O(n * (k-1)^(g/2)) which is manageable for n=1250, g=9.
    def check_girth(adj, n, g):
        for s in range(n):
            visited = set()
            queue = deque()
            for t in adj[s]:
                queue.append((s, t, 1))
                visited.add((s, t))
            while queue:
                u, prev, dist = queue.popleft()
                if dist >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and dist + 1 >= 3:
                        if dist + 1 < g:
                            return False
                    if (v, u) not in visited:
                        visited.add((v, u))
                        queue.append((v, u, dist + 1))
        return True

    if not check_girth(adj, luw_n, g):
        # Fallback: return empty if check fails (should not happen for LUW)
        return 0, []

    # 2. Search for a smaller graph using random regular graph generation
    # This combines the structural baseline of the LUW graph with the
    # refinement machinery of random search from the inspirations.
    
    # Moore bound (lower bound on n)
    r = (g - 1) // 2
    moore = 1
    for i in range(r):
        moore += k * (k - 1) ** i
    n_start = moore
    if n_start < 2 * k:
        n_start = 2 * k
    
    # We search for n < luw_n
    # The search space is huge, so we use a time-budgeted approach
    # with increasing n to find the smallest valid graph
    
    # Use a fast girth checker
    def girth_ok_fast(adj, g):
        n = len(adj)
        adj_t = [tuple(a) for a in adj]
        
        for s in range(n):
            # dist indexed by (prev, u) -> depth
            # Use a flat array for speed
            dist = np.full(n * n, -1, dtype=np.int8)
            queue = []
            
            for v in adj_t[s]:
                key = v * n + s
                dist[key] = 1
                queue.append((v, s))
            
            head = 0
            while head < len(queue):
                u, p = queue[head]
                head += 1
                d = dist[u * n + p]
                if d >= g - 1:
                    continue
                for w in adj_t[u]:
                    if w == p:
                        continue
                    if w == s:
                        return False
                    key = w * n + u
                    if dist[key] == -1:
                        dist[key] = d + 1
                        queue.append((w, u))
        return True
    
    def random_regular_graph(n, k, rng):
        """Generate a random k-regular graph on n vertices."""
        m = n * k
        stubs = np.repeat(np.arange(n, dtype=np.int64), k)
        perm = np.arange(m, dtype=np.int64)
        
        # Fisher-Yates shuffle
        for i in range(m - 1, 0, -1):
            j = rng() % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        
        adj = [[] for _ in range(n)]
        seen = [set() for _ in range(n)]
        
        for a in range(0, m, 2):
            u = int(stubs[perm[a]])
            v = int(stubs[perm[a + 1]])
            if u == v:
                return None
            if v in seen[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
            seen[u].add(v)
            seen[v].add(u)
        return adj
    
    # Search for smaller graphs
    # Start from the Moore bound and go up to luw_n - 1
    # Use a time budget for each n value
    
    best_n = luw_n
    best_edges = luw_edges
    
    # We'll search in increasing order of n
    # For each n, we try a limited number of random graphs
    
    # Time budget allocation
    # Give more time to smaller n values as they are more interesting
    
    for n in range(n_start, luw_n):
        if time.time() > deadline:
            break
        
        # Allocate time based on n
        # Smaller n gets more trials
        if n < 500:
            max_trials = 100
        elif n < 800:
            max_trials = 50
        elif n < 1000:
            max_trials = 20
        else:
            max_trials = 10
        
        # Check time for each trial
        trial_time_budget = (deadline - time.time()) / max_trials if max_trials > 0 else 0
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Generate random graph
            adj = random_regular_graph(n, k, np.random.default_rng)
            if adj is None:
                continue
            
            # Check girth
            if girth_ok_fast(adj, g):
                # Found a valid graph
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
    
    # If no smaller graph found, return the LUW graph
    return luw_n, luw_edges
