"""
Construct a k-regular graph with girth at least g.
For the specific case k=3, g=16, we search for a small circulant graph,
falling back to the Lazebnik-Ustimenko-Woldar construction.
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
    deadline = start_time + 110  # 110s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > deadline

    # For the specific case k=3, g=16:
    # We search for a circulant graph C_n(1, a, b) with girth >= g.
    # If found, it will be much smaller than LUW.
    
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
    # We limit n to a reasonable range to ensure we finish in time.
    # If we don't find one up to 1000, we give up.
    max_n = 1000
    
    for n in range(min_n, max_n + 1, 2): # n must be even for 3-regular? No, 3n must be even, so n must be even.
        if (time.time() - start_time) > 100:
            return None, None
        
        # Generate pairs (a, b)
        # 1 < a < b < n/2
        # Also need to ensure gcd(n, a, b) = 1 for connectedness? 
        # Actually, if gcd(n, 1, a, b) != 1, the graph is disconnected.
        # Since 1 is in the set, gcd(n, 1, a, b) is always 1. So it is connected.
        
        for a in range(2, n // 2 + 1):
            for b in range(a + 1, n // 2 + 1):
                # Quick check for 4-cycles
                # A 4-cycle exists if:
                # 1 + a = b
                # 1 + b = a
                # a + b = 1
                # 1 = a + b (mod n) -> a+b = n-1 or 1
                # a = b + 1
                # b = a + 1
                # 1 + a + b = 0 (mod n) -> a+b = n-1
                # 1 - a - b = 0 (mod n) -> a+b = 1
                # a - 1 - b = 0 (mod n) -> a-b=1
                # b - 1 - a = 0 (mod n) -> b-a=1
                # a - b + 1 = 0 (mod n) -> b-a=1
                # b - a + 1 = 0 (mod n) -> a-b=1
                
                # More simply:
                # Check if any sum of two distinct generators is another generator or 0 or -1?
                # Generators: S = {1, -1, a, -a, b, -b}
                # 4-cycle if x+y = z where x,y,z in S and x!=y.
                # Or x+y = 0 (backtrack, not a cycle).
                
                # Let's just do a BFS of depth 4 to be safe.
                # But that's expensive.
                
                # Heuristic: if a and b are small, likely short cycles.
                # We can skip if a+b < 16? No.
                
                # Let's do a fast BFS check for girth < 16.
                # Since it's vertex transitive, check from 0.
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
                        # Note: if shift == n-shift, i.e., 2*shift = n, we have a double edge?
                        # No, if 2*shift = n, then i -> i+shift -> i is a 2-cycle?
                        # But graph is simple.
                        # If 2*shift = n, then edge (i, i+shift) is the same as (i+shift, i).
                        # We only add if i < j.
                        # If 2*shift = n, then j = i + n/2.
                        # If i < i+n/2, we add it.
                        # Is it possible to add it twice?
                        # If shift1 = n/2 and shift2 = n/2? No, a,b distinct from 1.
                        # If a = n/2, then edge (i, i+n/2).
                        # If b = n/2? No, a < b.
                        # So each edge is added once.
                        
                return n, edges
    
    return None, None


def _has_short_cycle_circulant(n, a, b, g):
    """
    Check if the circulant graph C_n(1, a, b) has a cycle shorter than g.
    Uses BFS from 0.
    """
    # BFS
    visited = [-1] * n
    parent = [-1] * n
    visited[0] = 0
    queue = deque([(0, 0)]) # (node, depth)
    
    while queue:
        u, depth = queue.popleft()
        
        if depth >= g // 2:
            # We are checking for cycles of length < g.
            # A cycle of length L is found if we reach a visited node v
            # such that depth(u) + depth(v) + 1 >= L?
            # Actually, if we visit a node at depth d, and it was visited at depth d'
            # then we have a cycle of length d + d' + 1.
            # We want to know if min cycle length < g.
            # So if d + d' + 1 < g, return True.
            
            # But we can stop early if we don't find any short cycles.
            # We only need to explore up to depth g//2.
            pass
        
        for shift in [1, a, b]:
            v1 = (u + shift) % n
            v2 = (u - shift) % n
            
            for v in [v1, v2]:
                if visited[v] == -1:
                    visited[v] = depth + 1
                    parent[v] = u
                    queue.append((v, depth + 1))
                else:
                    # Found a cycle
                    # Cycle length is depth + 1 + visited[v]
                    # But this might be the edge we came from?
                    # No, BFS tree.
                    # If v is parent of u, skip?
                    # We don't store parent in queue, but we can check.
                    # Actually, if visited[v] < depth, it's a cross edge.
                    # Cycle length = depth + 1 + visited[v]
                    # But this assumes the path to v is disjoint from u except at v?
                    # In a tree, yes.
                    # However, this detects *some* cycle.
                    # We need to ensure it's a simple cycle.
                    # BFS detects the shortest path to each node.
                    # The cycle formed by two shortest paths to the same node is not necessarily simple?
                    # Actually, in a graph, the shortest cycle through a node is not necessarily the one found by merging two BFS paths.
                    # However, if the graph is bipartite, it's easier.
                    # Circulant graphs are not always bipartite.
                    
                    # A safer way:
                    # If we encounter a visited node v, the cycle length is depth + 1 + visited[v].
                    # This is valid if the path to v and path to u are disjoint?
                    # Not necessarily.
                    
                    # Let's just use the standard BFS girth algorithm:
                    # For each edge (u, v), if visited[u] != -1 and visited[v] != -1
                    # and u != parent[v] and v != parent[u], then cycle length is visited[u] + visited[v] + 1.
                    # We need to track parents.
                    
                    # Since this function is called many times, let's optimize.
                    # We'll just return False for now and rely on the fact that
                    # most random circulants have large girth?
                    # No, that's risky.
                    
                    # Let's implement a proper check.
                    # To save time, we'll only check if depth + visited[v] + 1 < g.
                    if depth + 1 + visited[v] < g:
                        # We must ensure it's not the immediate parent
                        # We don't have parent array here.
                        # Let's assume for now that if this condition holds, it's a short cycle.
                        # This might have false positives if v is the parent.
                        # But if v is the parent, visited[v] = depth - 1.
                        # Then depth + 1 + depth - 1 = 2*depth.
                        # If 2*depth < g, it's a valid cycle?
                        # No, that's a backtracking walk, not a cycle.
                        
                        # So we must skip if v is the parent.
                        # We can't do that easily without parent array.
                        
                        # Let's modify the BFS to store parent.
                        pass
        
        # Optimization: If we reach depth g//2 without finding a short cycle,
        # we might still have a cycle of length g//2 + 1?
        # We need to check all edges.
        
    # This implementation is incomplete/incorrect for girth.
    # Let's use a simpler heuristic or a more robust but slower check only for candidates that pass a quick filter.
    
    # Quick filter:
    # If a+b < g, likely short cycle?
    # Let's just return False to avoid rejecting good candidates due to buggy check,
    # and let the final verification (if we had one) handle it.
    # But we don't have a final verification in the current code structure.
    
    # Let's implement a correct but slightly slower BFS.
    return _bfs_girth_check(n, a, b, g)


def _bfs_girth_check(n, a, b, g):
    """
    Correct BFS girth check for circulant C_n(1, a, b).
    """
    visited = [-1] * n
    parent = [-1] * n
    visited[0] = 0
    queue = deque([(0, 0)])
    
    while queue:
        u, depth = queue.popleft()
        
        for shift in [1, a, b]:
            for v in [(u + shift) % n, (u - shift) % n]:
                if visited[v] == -1:
                    visited[v] = depth + 1
                    parent[v] = u
                    queue.append((v, depth + 1))
                else:
                    # Check if this forms a cycle
                    # If v is not the parent of u, and u is not the parent of v
                    if parent[v] != u and parent[u] != v:
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
