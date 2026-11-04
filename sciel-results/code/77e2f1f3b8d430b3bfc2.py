"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def _luw_graph(k, g):
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

    def _generate_random_3_regular(n):
        """Generate a random 3-regular simple graph using configuration model."""
        if n < 4 or n % 2 != 0:
            return None
        half_edges = []
        for i in range(n):
            for _ in range(3):
                half_edges.append(i)
        random.shuffle(half_edges)
        adj = {i: [] for i in range(n)}
        edges = []
        for i in range(0, n * 3, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
            edges.append((u, v))
        return edges, adj

    def _verify_girth(adj, n, g_limit):
        """Verify girth >= g_limit using non-backtracking BFS from each vertex."""
        # Non-backtracking BFS: from each vertex, walk up to depth g_limit//2
        # If we can return to start in < g_limit steps (non-backtracking), fail.
        # Actually, we need: no cycle of length < g_limit.
        # A cycle of length L exists iff there's a non-backtracking closed walk of length L.
        # For girth check: BFS from each vertex to depth g_limit//2 - 1.
        # If a vertex at depth d has an edge to a vertex at depth d' where d + d' + 1 < g_limit
        # and they're in the same BFS tree (not parent-child), we have a cycle.
        # Simpler: for each vertex, do BFS to depth floor((g_limit-1)/2).
        # If we find a back-edge creating a cycle of length < g_limit, fail.
        
        max_depth = (g_limit - 1) // 2
        for start in range(n):
            # BFS from start, tracking parent to avoid backtracking
            # dist[v] = shortest distance from start
            # We need to detect if there's a cycle of length < g_limit through start
            # Actually, any cycle in the graph has length < g_limit => fail
            # So: find the shortest cycle in the entire graph.
            # More efficient: BFS from each vertex, when we find an edge to a visited
            # vertex that isn't the parent, compute cycle length.
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and dist[u] + dist[v] + 1 < g_limit:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        # But this might not be the shortest cycle through start
                        # Actually, this detects ANY cycle of that length
                        return False
        return True

    def _verify_girth_rigorous(adj, n, g_limit):
        """Rigorous girth verification: check no cycle shorter than g_limit."""
        # For each vertex, BFS to find shortest path back to itself (non-trivial)
        # This is O(n * (n + m)) which is too slow for large n.
        # Better: use the fact that if girth >= g, then BFS tree from any vertex
        # to depth (g-1)//2 is a tree (no cycles in that depth).
        # If there's a cycle of length L < g, then from some vertex on that cycle,
        # BFS will find a back-edge within depth (g-1)//2.
        
        max_depth = (g_limit - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Back-edge found: cycle length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    # Try to find a smaller graph than LUW
    # Moore bound for k=3, g=16: 2*(2^0 + 2^1 + ... + 2^7) = 2*(2^8 - 1) = 510
    # Actually Moore bound for regular graph: 1 + k*sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i
    # For k=3, g=16: 1 + 3*(1 + 2 + 4 + 8 + 16 + 32 + 64) = 1 + 3*127 = 382
    # Wait, let me recalculate. Girth g=16 means no cycle of length < 16.
    # Moore bound: n >= 1 + k * sum_{i=0}^{(g/2-1)} (k-1)^i for even g
    # For g=16, k=3: sum_{i=0}^{7} 2^i = 255, so n >= 1 + 3*255 = 766
    # Hmm, that's the Moore bound. Let me use a safer lower bound.
    
    # Actually for the cage problem n(3,16), Moore bound is 512 (as stated in task).
    # Let's try sizes from 512 upward, in steps.
    
    # First, compute LUW to know the fallback size
    n_luw, edges_luw = _luw_graph(k, g)
    
    # Try randomized search for smaller graphs
    # Start from a reasonable lower bound and go up
    # For k=3, g=16, try from ~512 to n_luw-1
    # But verification is expensive, so limit attempts
    
    if k == 3 and g == 16:
        # Specific case: try to find graph smaller than LUW (1536)
        # Try sizes: 512, 520, 528, ..., up to some limit
        # Each verification is O(n * (n+m)) ~ O(n^2) for 3-regular
        # For n=512, that's ~262K ops per vertex * 512 vertices = too slow?
        # Actually BFS is O(n + m) = O(n) for 3-regular, so total O(n^2)
        # For n=512: 512^2 = 262K, fine. For n=1000: 1M, still fine.
        
        # Try a few random graphs at each size
        target_sizes = list(range(512, min(n_luw, 1000), 16))
        # Also try some larger sizes
        target_sizes += list(range(1000, n_luw, 32))
        
        for target_n in target_sizes:
            if time.time() > deadline - 5:
                break
            # Ensure target_n is even (needed for 3-regular)
            if target_n % 2 != 0:
                target_n += 1
            if target_n >= n_luw:
                break
            
            # Try up to 5 random graphs at this size
            max_attempts = 5
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                result = _generate_random_3_regular(target_n)
                if result is None:
                    continue
                edges, adj = result
                # Verify girth
                if _verify_girth_rigorous(adj, target_n, g):
                    return target_n, edges
    
    # Fallback to LUW
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
