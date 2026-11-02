import time
import sys
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Generate the seed's LUW graph as a baseline (Best Known: 936)
    n_luw, edges_luw = _construct_luw(k, g)
    
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Try to find a smaller graph using random search with fast NumPy girth check
    # Moore bound for 3-regular graph of girth g=16 is 510.
    # We search from 512 (even) up to best_n - 2.
    
    # If g is odd, the graph is not necessarily bipartite, but for k=3, g=16 is even.
    # We assume the search is for bipartite graphs if g is even.
    
    if g % 2 == 0:
        # Moore bound
        m = g // 2
        moore_bound = 2 * sum([(k-1)**i for i in range(m)])
        # Ensure even
        if moore_bound % 2 != 0:
            moore_bound += 1
    else:
        m = g // 2
        moore_bound = 1 + k * sum([(k-1)**i for i in range(m)])
        if moore_bound % 2 != 0:
            moore_bound += 1 # 3-regular graphs must have even number of vertices if bipartite, 
                             # but for odd girth, N can be odd? No, 3-regular graph must have even N?
                             # Sum of degrees = 3N. 3N must be even => N must be even.
                             # So N is always even for k=3.

    # Search range
    start_n = max(moore_bound, 4)
    
    # Limit the search to a reasonable range to avoid spending too much time
    # on sizes that are very unlikely to have high girth.
    # Girth 16 is very high. Random graphs of size 512 will almost certainly have small cycles.
    # We will try a few sizes and a few graphs per size.
    
    max_n_to_check = min(best_n - 2, start_n + 200) # Check up to 200 vertices above Moore bound
    
    if best_n > start_n:
        for n in range(start_n, max_n_to_check + 1, 2):
            if time.time() > deadline - 10: # Leave 10s for finalizing
                break
                
            # Try a few random graphs of this size
            # The number of attempts should be small because generation + check is not instantaneous
            attempts = 5
            for _ in range(attempts):
                if time.time() > deadline - 10:
                    break
                edges = _random_3_reg_graph(n)
                if edges is None:
                    continue
                if _has_girth_at_least_fast(n, edges, g):
                    return n, edges
                    
    return best_n, best_edges

def _construct_luw(k, g):
    # Lazy-evaluated LUW construction from seed
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

def _random_3_reg_graph(n):
    """Generate a random simple 3-regular graph of n vertices using configuration model.
    Returns None if impossible or if multi-edges/loops are generated."""
    if n < 4 or n % 2 != 0:
        return None
    # Configuration model
    # Each vertex has 3 stubs
    stubs = [i for i in range(n) for _ in range(3)]
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.add((u, v))
    return list(edges)

def _has_girth_at_least_fast(n, edges, g):
    """Check if graph has no cycle shorter than g using NumPy matrix powers.
    For a bipartite graph (which we assume for even g), we check for cycles of length 4, 6, ..., g-2.
    A cycle of length 2k exists if the diagonal of A^(2k) is non-zero.
    """
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    # We need to check for cycles of length 4, 6, ..., g-2
    # Since g is even, let g = 2m. We check 2, 4, ..., 2m-2.
    # Cycle of length 2 (multi-edge) is already excluded by simple graph check.
    # So we start from 4.
    
    # A^2
    # If any diagonal element of A^2 is > 0, there is a cycle of length 2 (impossible in simple graph) 
    # or we are checking for triangles? No, A^2[i,i] counts paths of length 2 from i to i.
    # In a simple graph, A^2[i,i] = degree(i). This is not a cycle.
    # A cycle of length L corresponds to closed walks of length L.
    # For simple graphs, the number of cycles of length L is related to trace(A^L).
    # Specifically, trace(A^L) counts closed walks.
    # For L=4, trace(A^4) counts closed walks of length 4.
    # A closed walk of length 4 in a simple graph is either a 4-cycle or a "back-and-forth" on two edges (i-j-i-k-i? No, i-j-i is length 2).
    # Actually, for simple graphs, if trace(A^L) > 0, it doesn't guarantee a simple cycle of length L, 
    # but for L=4, a closed walk i-j-k-j-i is length 4 but not a simple cycle.
    # However, if the graph has girth >= g, then there are NO closed walks of length < g that are not "backtracking".
    # This method is tricky.
    
    # Better: BFS is safer for small N.
    # But let's try the matrix method for speed.
    # Actually, for N=512, BFS is $512 \times 512 \approx 260k$ steps. 
    # In Python, 260k steps is ~0.1s. 
    # 5 attempts * 50 sizes = 250 checks * 0.1s = 25s. 
    # This fits in the budget!
    
    # So we can just use the standard BFS girth checker, but optimized.
    return _has_girth_at_least_bfs(n, edges, g)

def _has_girth_at_least_bfs(n, edges, g):
    """Standard BFS girth check, optimized with early exit."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # This cycle contains the start vertex.
                    # If this is the shortest cycle containing start, and we check all starts,
                    # we find the girth.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
