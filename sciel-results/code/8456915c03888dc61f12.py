"""
Construct a k-regular graph with girth at least g.
Optimized search for circulant graphs C_n(1, a, b) with fallback to LUW.
"""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin out of 150s limit
    
    def check_deadline():
        return time.time() > deadline

    # For the specific case k=3, g=16:
    # We search for a circulant graph C_n(1, a, b) with girth >= g.
    if k == 3 and g == 16:
        try:
            n_circ, edges_circ = _find_circulant_cage(3, 16, deadline)
            if n_circ is not None:
                return n_circ, edges_circ
        except Exception:
            pass
    
    # Fallback: Lazebnik-Ustimenko-Woldar
    return _luw_construct(k, g)


def _find_circulant_cage(k, g, deadline):
    """
    Search for a circulant graph C_n(1, a, b) with girth at least g.
    Assumes k=3.
    """
    start_time = time.time()
    
    # Moore bound for k=3, g=16 (even girth):
    # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * (2^8 - 1) = 510
    # n must be even. So start from 512.
    min_n = 512
    
    # Search space
    # We limit n to a reasonable range.
    # LUW gives 1536. We want to find something smaller.
    # Let's search up to 1200 first. If not found, expand? 
    # No, better to be safe. 1200 is a good upper bound for a quick search.
    max_n = 1200
    
    for n in range(min_n, max_n + 1, 2): 
        if (time.time() - start_time) > 105:
            return None, None
        
        # Generate pairs (a, b)
        # 1 < a < b < n/2
        # To ensure connectivity and simple graph, gcd(n, 1, a, b) must be 1.
        # Since 1 is in the set, gcd is always 1.
        
        for a in range(2, n // 2):
            if (time.time() - start_time) > 105:
                return None, None
                
            for b in range(a + 1, n // 2):
                # Quick heuristic filters to avoid BFS for obviously bad candidates
                # 1. a+b should not be small or equal to 1, a, b, n-1, n-a, n-b
                # 2. b-a should not be small or equal to 1, a, b, n-1, n-a, n-b
                
                # Let's do a very fast check for 4-cycles and 6-cycles first?
                # Or just run the BFS if we are within time.
                
                # Optimization: Check if a, b are "compatible"
                # If a+b == 1 % n -> 4-cycle
                # If b-a == 1 % n -> 4-cycle
                # If a+b == a % n -> b=0 (impossible)
                # If a+b == b % n -> a=0 (impossible)
                # If a+b == n-1 % n -> 4-cycle (1 + (n-1-a) = n-a? No. 1+a+b = n -> 1+a+b-1 = n-1. Cycle 0-1-(1+a)-b? No.)
                
                # Let's rely on the BFS check which is robust.
                # But we can skip if a+b < 4 or similar trivial cases, though a>=2, b>=3 so a+b>=5.
                
                if _has_short_cycle_circulant(n, a, b, g):
                    continue
                
                # If we pass the girth check, we have a candidate.
                # Construct edges
                edges = []
                for i in range(n):
                    for shift in [1, a, b]:
                        j = (i + shift) % n
                        if i < j:
                            edges.append((i, j))
                        
                return n, edges
    
    return None, None


def _has_short_cycle_circulant(n, a, b, g):
    """
    Check if the circulant graph C_n(1, a, b) has a cycle shorter than g.
    Uses BFS from 0.
    Returns True if a short cycle is found, False otherwise.
    """
    # We only need to check girth from vertex 0 due to vertex transitivity.
    # A cycle of length L < g exists if there is an edge (u, v) such that
    # dist(u) + dist(v) + 1 < g, where dist is from source 0.
    # We can stop BFS when depth >= g//2.
    
    visited = [-1] * n
    parent = [-1] * n
    visited[0] = 0
    queue = deque([(0, 0)]) # (node, depth)
    
    max_depth = g // 2
    
    while queue:
        u, depth = queue.popleft()
        
        # If we have reached max depth, we don't need to expand further
        # because any cycle found involving a node at depth > max_depth
        # would have length > 2*max_depth = g (for even g) or similar.
        # Actually, if depth >= max_depth, we still need to check edges from u
        # to see if they connect to a node with small depth.
        # But we don't enqueue neighbors if depth + 1 > max_depth?
        # If we don't enqueue, we might miss a cycle where one end is deep and other is shallow?
        # No, if one end is deep (>= g/2) and the other is shallow, the sum might be < g?
        # Example: g=16. max_depth=8.
        # Edge (u, v) with dist(u)=8, dist(v)=1. Cycle len = 8+1+1=10 < 16.
        # So we MUST check edges from nodes at depth max_depth.
        # But we don't need to enqueue nodes at depth max_depth + 1.
        
        for shift in [1, a, b]:
            v1 = (u + shift) % n
            v2 = (u - shift) % n
            
            for v in [v1, v2]:
                if visited[v] == -1:
                    visited[v] = depth + 1
                    parent[v] = u
                    if depth + 1 <= max_depth:
                        queue.append((v, depth + 1))
                else:
                    # If v is already visited, check for cycle
                    # Skip if v is the parent of u (backtrack)
                    if parent[v] == u or parent[u] == v:
                        continue
                    
                    cycle_len = depth + 1 + visited[v]
                    if cycle_len < g:
                        return True
                        
    return False

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make it odd
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

# EVOLVE-BLOCK-END
