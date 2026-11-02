"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if k == 4 and g == 10:
        # Target n values to try, in increasing order
        target_ns = [162, 164, 166, 168, 170, 172, 174, 176, 178, 180,
                     182, 184, 186, 188, 190, 192, 194, 196, 198, 200,
                     204, 208, 212, 216, 220, 224, 228, 232, 236, 240,
                     244, 248, 252, 256, 260, 264, 268, 272, 276, 280,
                     284, 288, 292, 296, 300, 304, 308, 312, 316, 320]
        
        for n in target_ns:
            if time.time() > deadline - 3:
                break
            result = _random_search(n, k, g, deadline)
            if result is not None:
                return result
        
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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


def _random_search(n, k, g, deadline):
    """Try to find a random k-regular graph with girth >= g on n vertices."""
    max_attempts = 200
    
    for _ in range(max_attempts):
        if time.time() > deadline - 3:
            return None
        
        edges = _gen_random_4reg(n)
        if edges is None:
            continue
        
        if _has_girth_at_least(edges, n, g):
            return n, edges
            
    return None


def _gen_random_4reg(n):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4:
        return None
    if n * 4 % 2 != 0:
        return None
    
    # Use configuration model with rejection
    for attempt in range(30):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                valid = False
                break
            edges.add((u, v))
        
        if valid:
            return list(edges)
    
    return None


def _has_girth_at_least(edges, n, g):
    """Check if graph has girth at least g. Returns True if no cycle shorter than g exists."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each start vertex, do BFS to depth g//2 and check all non-tree edges
    # for creating cycles shorter than g.
    max_depth = g // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
        
        # Check all edges for cycles shorter than g
        for u, v in edges:
            if dist[u] != -1 and dist[v] != -1:
                # Only need to check if this edge could form a cycle < g
                # The cycle through this edge has length dist[u] + dist[v] + 1
                # But we must ensure u and v are in the same "branch" or connected
                # Actually, for any edge (u,v), if both are reachable from start,
                # the cycle formed by the BFS path to u, the edge (u,v), and the BFS path to v
                # has length dist[u] + dist[v] + 1 only if the paths to u and v from start
                # share no vertices other than start (which is guaranteed by BFS tree).
                # Wait, that's not right. The BFS tree paths may share vertices.
                # The correct check: for each non-tree edge (u,v), the cycle length is
                # dist[u] + dist[v] + 1 - 2*dist[lca(u,v)]. But we don't track LCA.
                # 
                # Simpler correct approach: for each vertex u, for each neighbor v != parent[u],
                # if v is visited and dist[u] + dist[v] + 1 < g, we have a cycle < g.
                # But this double-counts. Let's just check: for each edge (u,v) that is NOT
                # a tree edge, if dist[u] != -1 and dist[v] != -1 and dist[u] + dist[v] + 1 < g,
                # then there's a cycle of length at most dist[u] + dist[v] + 1 < g.
                # This is a sufficient condition (not necessary in all cases, but combined
                # over all starts it becomes necessary).
                #
                # Actually the standard result: girth >= g iff for every vertex s, 
                # for every edge (u,v) not in the BFS tree from s, dist_s[u] + dist_s[v] + 1 >= g.
                # This is because any cycle of length L < g, picking s on the cycle, the two
                # tree paths from s to the endpoints of any non-tree edge on the cycle give
                # a cycle of length <= L.
                
                # We need to skip tree edges. An edge (u,v) is a tree edge if parent[u]==v or parent[v]==u.
                if parent[u] == v or parent[v] == u:
                    continue
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g:
                    return False
    
    return True
# EVOLVE-BLOCK-END
