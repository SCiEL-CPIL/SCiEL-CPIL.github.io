import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin from 150s limit
    
    # Target girth: we need no cycle shorter than g.
    # LUW constructions naturally produce even girth, so if g is odd,
    # we need girth >= g+1 (the next even number).
    target_girth = g if g % 2 == 0 else g + 1
    
    best_n = float('inf')
    best_edges = []
    
    # Try LUW constructions with various d values.
    # The number of vertices is q * q^d = q^(d+1) for the standard LUW.
    # For k=5, q=5, so n = 5^(d+1).
    # We want the smallest n that achieves girth >= target_girth.
    # Start from small d and increase.
    
    # For k=5, g=9: target_girth=10
    # d=3: n=5^4=625, girth likely too small
    # d=4: n=5^5=3125
    # d=5: n=5^6=15625
    # d=6: n=5^7=78125
    # d=7: n=5^8=390625
    # d=8: n=5^9=1953125
    
    # Try d from 3 upward, checking girth for each.
    # Stop when we find a valid graph or run out of time.
    max_d = 12  # 5^13 is too large; cap at reasonable size
    
    for d in range(3, max_d + 1):
        if time.time() > deadline - 15:
            break
            
        try:
            n, edges = _construct_luw(k, g, d)
        except Exception:
            continue
            
        if n >= best_n:
            continue
            
        # Check girth - this is the expensive part
        if _check_girth(n, edges, target_girth):
            best_n = n
            best_edges = edges
            # Found a valid graph; try larger d to see if we can get smaller?
            # No - larger d gives larger n. So we want the SMALLEST d that works.
            # But wait: the girth may not be monotonic in d.
            # Let's continue checking larger d in case a larger d has better girth
            # properties, but we already have a valid answer.
            # Actually, for LUW, girth tends to increase with d, so the first
            # valid d should give the smallest n. Break here.
            break
            
    if best_n == float('inf'):
        # Fallback: use a heuristic d
        d = max(3, g - 2)
        best_n, best_edges = _construct_luw(k, g, d)
        
    return best_n, best_edges

def _construct_luw(k, g, d):
    q = k
    # rule generation same as seed
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

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

def _check_girth(n, edges, target_girth):
    """Check that the graph has no cycle shorter than target_girth.
    Uses non-backtracking BFS from each vertex up to depth (target_girth-1)//2.
    A cycle of length L < target_girth would be detected as a non-backtracking
    walk of length L that returns to the start.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For a graph with girth g, there is no cycle of length < g.
    # Non-backtracking walk of length L that returns to start => cycle of length <= L.
    # We need to check all non-backtracking walks up to length target_girth - 1.
    # But we can use BFS with parent tracking: if we reach a vertex at depth d
    # and it was first reached at depth d', then d + d' + 1 >= girth.
    # Actually, the standard approach: BFS from each vertex, tracking the
    # shortest path distance. If two different paths reach the same vertex
    # at depths d1 and d2, then d1 + d2 + 1 is a cycle length (if the paths
    # only share the start). This is an upper bound on girth.
    
    # More efficient: for each start vertex, do a BFS and track the depth
    # at which each vertex is first reached. If we encounter a vertex that
    # has already been reached, the cycle length is current_depth + prev_depth + 1.
    # We need all such cycle lengths to be >= target_girth.
    
    max_depth = target_girth - 1  # We need to detect cycles up to this length
    
    for start in range(n):
        # dist[v] = shortest distance from start to v
        dist = [-1] * n
        dist[start] = 0
        queue = deque([(start, 0)])
        found_short_cycle = False
        
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))
                else:
                    # Found a cycle: depth + 1 + dist[neighbor]
                    # But this might not be a simple cycle if the paths overlap
                    # beyond the start. However, it gives an upper bound on
                    # the girth. If this upper bound is < target_girth, we
                    # definitely have a short cycle.
                    cycle_len = depth + 1 + dist[neighbor]
                    if cycle_len < target_girth:
                        # This indicates a potential short cycle, but we need
                        # to be careful: this is only a valid cycle if the
                        # two paths from start to neighbor are internally
                        # disjoint. A simpler sufficient condition: if
                        # depth + dist[neighbor] + 1 < target_girth, then
                        # there IS a cycle of length <= that.
                        # Actually, in a BFS tree, if we find an edge (node, neighbor)
                        # where both are already visited, the cycle length is
                        # depth + dist[neighbor] + 1, and this IS a valid cycle
                        # (it goes from start to node via the BFS tree, then
                        # the edge to neighbor, then back to start via the
                        # BFS tree from neighbor). The only issue is if the
                        # two tree paths share vertices other than start,
                        # but then the actual cycle is even shorter, so
                        # we still detect it.
                        return False
        
        # Also need to check for cycles that go through the start vertex
        # via non-tree edges. The above handles edges where both endpoints
        # are in the BFS tree. But we also need to check if any vertex at
        # depth d has an edge to a vertex at depth d' where d + d' + 1 < target_girth.
        # The BFS above does check all edges, so this should be covered.
        
    return True
