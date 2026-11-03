"""Program for constructing small k-regular graphs of given girth."""
import numpy as np
import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using a greedy backtracking search
    with bounded-depth BFS for cycle checking.
    """
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        return 2, [(0, 1)]
    
    # Moore Bound
    if k == 2:
        moore_bound = g
    else:
        # n >= 2 * sum_{i=0}^{(g-1)/2} (k-1)^i
        # For g even, D = g/2 - 1. For g odd, D = (g-1)/2.
        # Actually, the standard Moore bound for girth g:
        # If g is even, g=2D, n >= 2 * sum_{i=0}^{D-1} (k-1)^i
        # If g is odd, g=2D+1, n >= k * sum_{i=0}^{D-1} (k-1)^i
        if g % 2 == 0:
            D = g // 2
            # Sum i=0 to D-1
            s = 0
            term = 1
            for _ in range(D - 1):
                s += term
                term *= (k - 1)
            moore_bound = 2 * s
        else:
            D = (g - 1) // 2
            s = 0
            term = 1
            for _ in range(D):
                s += term
                term *= (k - 1)
            moore_bound = k * s
            
    # We start from the Moore bound and increase n if we fail to find a solution
    # to keep the score (minimize n) as low as possible.
    
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer
    
    # Greedy Backtracking Search
    # We will try different n values starting from moore_bound
    
    # Optimization: Use a simple BFS with depth limit to check for short cycles.
    def has_short_cycle(adj, u, v, max_depth):
        """
        Check if there is a path from u to v of length < max_depth in the graph defined by adj.
        Note: We are checking if adding edge (u, v) creates a cycle < g.
        This is true if there is a path from u to v of length < g-1.
        So we set max_depth = g-2 (we want to find a path of length <= g-2).
        If found, return True.
        """
        if u == v:
            return False # No self-loops allowed, but we check distinct u, v
        
        visited = {u: 0}
        queue = deque([(u, 0)])
        
        while queue:
            node, dist = queue.popleft()
            if dist > max_depth:
                continue
            if node == v:
                return True
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return False

    def try_construct(n, k, g, seed=0):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = []
        
        # We need to add edges such that every vertex has degree k.
        # Order of vertices: randomize to avoid symmetries?
        # Or just go 0 to n-1.
        
        # To make it faster, we can pre-determine the order in which we assign neighbors.
        # We'll iterate through vertices.
        
        # Backtracking stack
        # State: current vertex index, list of chosen neighbors for this vertex
        # This is complex. Instead, use a simpler recursive function with restarts.
        
        def backtrack(vertex_idx, adj, edges):
            if vertex_idx == n:
                return True
            
            # Determine how many more neighbors this vertex needs
            current_deg = len(adj[vertex_idx])
            needed = k - current_deg
            
            if needed <= 0:
                return backtrack(vertex_idx + 1, adj, edges)
            
            # Candidates: all vertices j > vertex_idx? 
            # No, we can connect to any j != vertex_idx.
            # But to avoid duplicates, we usually only consider j > vertex_idx OR
            # we manage a global pool.
            # Let's just pick from all vertices not yet connected, ensuring we don't exceed their degree.
            
            candidates = []
            for j in range(n):
                if j == vertex_idx:
                    continue
                if len(adj[j]) < k:
                    # Check if edge (vertex_idx, j) is already present
                    if j not in adj[vertex_idx]:
                        candidates.append(j)
            
            random.shuffle(candidates)
            
            for j in candidates:
                # Check if adding (vertex_idx, j) creates a short cycle
                # We need to check if there is a path from vertex_idx to j of length < g-1
                # in the current graph.
                # Note: The current graph includes edges added so far.
                
                if has_short_cycle(adj, vertex_idx, j, g - 2):
                    continue
                
                # Add edge
                adj[vertex_idx].add(j)
                adj[j].add(vertex_idx)
                edges.append((vertex_idx, j))
                
                # Recurse
                if backtrack(vertex_idx, adj, edges):
                    return True
                
                # Backtrack
                adj[vertex_idx].remove(j)
                adj[j].remove(vertex_idx)
                edges.pop()
            
            return False
        
        return backtrack(0, adj, edges), edges

    # Try increasing n
    n = moore_bound
    while True:
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random seeds for this n
        success = False
        for seed in range(10):
            if time.time() - start_time > time_limit:
                break
            ok, edges = try_construct(n, k, g, seed)
            if ok:
                success = True
                break
        
        if success:
            return n, edges
        
        # If failed, increase n
        # Increase by a small amount to explore the space
        n += 2  # Must keep parity? Regular graphs of odd degree require even n.
                # If k is odd, n must be even.
                # If k is even, n can be any.
                # For k=3, n must be even.
                # Moore bound for k=3, g=16 is 510 (even).
                # So step by 2 is safe.
        
        # If n gets too large, we might time out.
        # But we must return something.
        # If we run out of time, we return the best we have?
        # The task requires a valid graph.
        # If we fail to find one, we should probably return a known valid graph
        # or the LUW construction as a fallback?
        # The directive says "design a new program...".
        # If the greedy fails, I can fall back to LUW to ensure validity.
        
    # Fallback to LUW if greedy fails (to ensure valid output)
    # Re-implement LUW concisely for fallback
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
        
    # LUW Construction for fallback
    d_odd = g - 5
    if d_odd < 3: d_odd = 3
    if d_odd % 2 == 0: d_odd += 1
    d_even = g - 4
    if d_even < 2: d_even = 2
    if d_even % 2 == 1: d_even += 1
    d = min(d_odd, d_even)
    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
        
    rule = {}
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
            yield (1 - side, tuple(y))
            
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
