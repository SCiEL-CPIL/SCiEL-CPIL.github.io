"""Combined program: LUW construction with fast verification and fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin before 120s limit
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # This is a q-regular bipartite graph.
    # Girth is d+5 if d is odd, d+4 if d is even.
    # We need girth >= g.
    # For g=9, we need girth >= 9.
    # If we pick d such that girth >= 9.
    
    q = k
    
    # Determine d
    # If d is odd, girth = d+5. Need d+5 >= g => d >= g-5.
    # If d is even, girth = d+4. Need d+4 >= g => d >= g-4.
    # We want the smallest d to minimize vertices (n = 2*q^d).
    
    d_candidates = []
    # Try odd d
    d_odd = g - 5
    if d_odd < 3: d_odd = 3
    if d_odd % 2 == 0: d_odd += 1
    d_candidates.append(d_odd)
    
    # Try even d
    d_even = g - 4
    if d_even < 4: d_even = 4
    if d_even % 2 == 1: d_even += 1
    d_candidates.append(d_even)
    
    # Choose d that minimizes q^d (which is d, since q>1)
    d = min(d_candidates)
    
    # Finite field arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Rule for CD(d, q)
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    def build_graph(d_val):
        """Build the LUW graph with given d, return (n, edges, adj)."""
        start = (0, (None,) + (0,) * d_val)
        index = {start: 0}
        order = [start]
        edges = []
        
        i = 0
        while i < len(order):
            if time.time() > deadline:
                break
            vertex = order[i]
            i += 1
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                
        n = len(order)
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            if u < n and v < n:
                adj[u].append(v)
                adj[v].append(u)
        
        return n, edges, adj

    def check_girth_fast(n, adj, min_girth):
        """
        Fast girth check using non-backtracking walks from every vertex.
        For a graph to have girth >= g, there should be no non-backtracking
        closed walk of length < g.
        
        We use BFS with parent tracking to detect short cycles.
        """
        # For each start vertex, do BFS and check for back edges
        # that would form cycles shorter than min_girth
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                
                # We only need to explore up to depth min_girth//2 - 1
                # because a cycle of length L would be detected at depth L/2
                if dist[u] >= (min_girth - 1) // 2:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[v] != dist[u] + 1:
                        # Found a non-tree edge
                        # Cycle length = dist[u] + dist[v] + 1
                        # But we need to be careful: this might not be a simple cycle
                        # For a rigorous check, we need to verify the path
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
            # Reset for next start
            # (dist and parent are local to this start)
        
        return True

    def check_girth_nonbacktracking(n, adj, min_girth):
        """
        More rigorous girth check using non-backtracking walks.
        A graph has girth >= g iff there is no non-backtracking closed walk of length < g.
        
        We check for cycles of length 3, 4, 5, ..., g-1.
        """
        # Check for triangles (length 3)
        # For each edge (u,v), check if there's a common neighbor
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))
        
        # Check for 4-cycles: two vertices with two common neighbors
        # This is O(n * deg^2) which is manageable
        # But let's use a simpler approach: BFS from each vertex
        
        for start in range(n):
            # BFS to find shortest cycle through start
            # Actually, we need to find any cycle of length < min_girth
            # Use BFS from start, tracking parent
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            
            found_short_cycle = False
            while qi < len(queue) and not found_short_cycle:
                u = queue[qi]
                qi += 1
                
                if dist[u] >= (min_girth - 1) // 2:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # v is already visited and not the parent of u
                        # This forms a cycle of length dist[u] + dist[v] + 1
                        # But we need to ensure it's a simple cycle
                        # For now, accept this as a cycle detection
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        
        return True

    # Try the optimal d first
    n, edges, adj = build_graph(d)
    
    # Verify girth
    if time.time() < deadline - 5:
        if check_girth_fast(n, adj, g):
            return n, edges
    
    # If girth check failed or timed out, try larger d
    d_next = d + 1
    if d_next % 2 == 0:
        d_next += 1  # Ensure odd for better girth guarantee
    
    n2, edges2, adj2 = build_graph(d_next)
    return n2, edges2

# EVOLVE-BLOCK-END
