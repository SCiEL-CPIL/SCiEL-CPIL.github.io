import time
import numpy as np
from collections import deque

def _build_luw(d, q):
    """
    Build the Lazebnik-Ustimenko-Woldar graph CD(d, q).
    n = 2 * q^d, degree = q, girth >= 2d+4 (for d>=3).
    Vertices: (side, x) where side in {0,1}, x in GF(q)^d.
    """
    q_int = int(q)
    
    # Rule for computing y[c-1] from previous y values and x
    # For c in 2..d:
    #   y[c-1] = x[c-1] + y[a-1] * x[b-1]  (side 0)
    #   y[c-1] = x[c-1] - x[a-1] * y[b-1]  (side 1)
    # where (a, b) = rule[c]
    
    rule = {}
    # Standard LUW rules for girth maximization
    # For d=3: rule[2]=(1,1), rule[3]=(2,1)
    # For d=4: rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2)
    # For d=5: rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2), rule[5]=(1,3)
    # General pattern:
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2)
    # c=5: (1,3)
    # c=6: (2,3)? or (1,4)?
    # Let's use the pattern from literature:
    # The rule is typically y_j = x_j + y_{j-1} * x_{j-2} for odd j, etc.
    # A common valid rule set for CD(d,q):
    # a(c) = c-1 if c is even, c-2 if c is odd?
    # b(c) = 1 if c is even, c-2 if c is odd?
    # Let's stick to a known valid sequence for small d.
    
    # Known valid rules for d up to 5:
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    if d >= 5: rule[5] = (1, 3)
    if d >= 6: rule[6] = (2, 3)
    if d >= 7: rule[7] = (1, 4)
    
    # If d is larger, extend with a pattern that maintains girth.
    # A simple pattern that works for many cases:
    # For c >= 2:
    # a = c - 1
    # b = 1 if c % 2 == 0 else c - 2
    # Let's verify this pattern:
    # c=2: a=1, b=1 (even c -> b=1). Matches (1,1).
    # c=3: a=2, b=1 (odd c -> b=c-2=1). Matches (2,1).
    # c=4: a=3, b=1 (even c -> b=1). Wait, previous was (1,2).
    # The pattern a=c-1, b=1 for even c gives (3,1) for c=4.
    # The previous explicit rule was (1,2).
    # Different rules give different girths. The explicit ones are known to be good.
    # For c > 7, let's use the a=c-1, b=1 pattern as a fallback, or just stop at d=7.
    # Since we only need d=3 for the best result, and maybe d=4,5 as fallback,
    # the explicit rules up to 5 are sufficient.
    
    for c in range(6, d + 1):
        # Use a generic pattern
        if c % 2 == 0:
            rule[c] = (c - 1, 1)
        else:
            rule[c] = (c - 1, c - 2)
            
    # Precompute neighbor relations
    # Vertices are indexed 0 to 2*q^d - 1.
    # Index = side * q^d + sum(x_i * q^(d-1-i))
    
    qd = q_int ** d
    n = 2 * qd
    adj = [[] for _ in range(n)]
    edges = []
    
    # Precompute powers of q for index conversion
    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i - 1] * q_int
    
    # Helper to convert tuple to index
    def tuple_to_idx(x):
        val = 0
        for i in range(d):
            val += x[i] * q_powers[d - 1 - i]
        return val
    
    # We only need to generate edges from side 0 to side 1
    # For each x in GF(q)^d:
    #   For each y_0 in GF(q):
    #     Compute y_1...y_{d-1}
    #     This defines a neighbor (1, y) for (0, x).
    
    # Let's iterate over all x in side 0
    for x_val in range(qd):
        # Convert x_val to tuple
        x = [0] * d
        rem = x_val
        for i in range(d - 1, -1, -1):
            x[i] = rem % q_int
            rem //= q_int
        
        for y0 in range(q_int):
            y = [0] * d
            y[0] = y0
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c - 1] = (x[c - 1] + y[a - 1] * x[b - 1]) % q_int
            
            y_val = 0
            for i in range(d):
                y_val += y[i] * q_powers[d - 1 - i]
            
            y_idx = qd + y_val
            x_idx = x_val
            
            adj[x_idx].append(y_idx)
            adj[y_idx].append(x_idx)
            edges.append((x_idx, y_idx))
            
    return n, edges, adj

def _girth_check(adj, n, g):
    """
    Check if the graph has girth at least g using non-backtracking walks.
    Returns True if girth >= g, False otherwise.
    Uses the fact that girth < g iff there exists a non-backtracking closed walk of length < g.
    For a k-regular graph, we can do BFS in the line graph or use the standard
    "for each directed edge, do non-backtracking BFS" approach.
    """
    # For girth >= g, we need no cycle of length < g.
    # A cycle of length L < g means there's a non-backtracking closed walk of length L.
    # We check: for each directed edge (u,v), do a non-backtracking BFS up to depth g-2.
    # If we reach a vertex w such that (v,w) is a valid non-backtracking continuation
    # and w == u (closing the walk), we found a cycle of length <= g-1.
    # More precisely: start with directed edge (s, t). At each step, extend to (t, u) where u != s.
    # If we ever reach (v, s) where (s,t) was the start, we have a cycle of length = path_length + 1.
    
    # Simpler standard approach: for each vertex s, do BFS with parent tracking.
    # If we find a back-edge (u,v) where v is an ancestor of u (not just parent),
    # the cycle length is depth[u] - depth[v] + 1.
    # We need this to be >= g for all such cycles.
    
    # Even simpler and correct: for each start s, BFS. When we see an already-visited
    # vertex v (not the parent), the cycle length is depth[u] + depth[v] + 1.
    # But this only finds the *shortest* cycle through s. We need ALL cycles.
    
    # Correct approach: for each vertex s, do BFS. For each edge (u,v) in the BFS tree
    # or non-tree edge, if both u and v are visited, the cycle length is
    # depth[u] + depth[v] + 1 - 2*depth[lca(u,v)]. But computing LCA is complex.
    
    # The standard efficient check: for each vertex s, do BFS. Maintain depth and parent.
    # For each non-tree edge (u,v) where depth[u] <= depth[v], the cycle length is
    # depth[u] + depth[v] + 1. But this underestimates if LCA is not s.
    
    # Actually, the BFS from s finds the shortest path from s to each vertex. 
    # For any cycle C, if s is on C, then the two paths from s to the two neighbors
    # of s on C give a cycle. If s is not on C, BFS won't directly detect it.
    
    # The correct method: for each vertex s, do BFS. For each edge (u,v) not in the tree,
    # the cycle length is depth[u] + depth[v] + 1. This gives a cycle that goes through s
    # only if the LCA of u and v in the BFS tree is s. Otherwise, it's a different cycle.
    
    # To be safe and correct, let's use: for each vertex s, BFS. For each non-tree edge (u,v),
    # compute cycle_len = depth[u] + depth[v] + 1. If cycle_len < g, return False.
    # This is conservative (may overestimate cycle length if LCA != s), but if it passes,
    # all cycles through s have length >= g. Since every cycle passes through some vertex,
    # this is correct.
    
    for s in range(n):
        depth = [-1] * n
        parent = [-1] * n
        depth[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if depth[v] == -1:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Non-tree edge or cross edge
                    cycle_len = depth[u] + depth[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    deadline = time.time() + 115.0
    best_n = float('inf')
    best_edges = None
    best_adj = None
    
    q = int(k)
    
    # Try d=3 first (n=250 for k=5)
    # Girth of CD(3, q) is at least 10. So it satisfies g=9.
    # Try d=4 (n=1250)
    # Girth of CD(4, q) is at least 12.
    
    for d in [3, 4, 5, 6]:
        if time.time() > deadline - 15:
            break
            
        try:
            n, edges, adj = _build_luw(d, q)
        except Exception:
            continue
            
        # Check validity
        # 1. Regularity
        valid = True
        for i in range(n):
            if len(adj[i]) != k:
                valid = False
                break
        if not valid:
            continue
            
        # 2. Girth
        if _girth_check(adj, n, g):
            if n < best_n:
                best_n = n
                best_edges = edges
                best_adj = adj
            break  # Smaller d is better, so stop if valid
            
    if best_edges is None:
        # Fallback: return a trivial invalid result or raise
        # This should not happen for k=5, g=9
        raise ValueError("No valid graph found")
        
    return best_n, best_edges
