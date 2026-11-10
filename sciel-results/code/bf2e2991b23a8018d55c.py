import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    if k != 4:
        return luw_construct(k, g)

    # Search for circulant graph C_n(a, b)
    # Steps: a, -a, b, -b
    # We want to minimize n.
    # Lower bound for (4,10) is 243.
    # We will search from 243 upwards.
    
    # Helper: Check if circulant graph with steps a, b on n vertices has girth >= g
    def check_girth_circulant(n, a, b, g_target):
        # Fast rejection: Check shortest cycle through 0
        # BFS from 0
        dist = [-1] * n
        dist[0] = 0
        q = deque([0])
        min_cycle_0 = float('inf')
        
        while q:
            u = q.popleft()
            du = dist[u]
            # Neighbors of u: u+a, u-a, u+b, u-b
            for v in ((u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n):
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)
                else:
                    # If v is already visited and not the parent, we found a cycle
                    # In BFS, the first time we see a visited node (that isn't the immediate parent)
                    # we have a candidate cycle.
                    # However, simple BFS tree doesn't directly give shortest cycle through root easily
                    # without checking all edges.
                    # A better way for "shortest cycle through 0" is to run BFS and for every edge (u,v),
                    # if dist[u] + dist[v] + 1 < current_min, update.
                    # But we need to be careful not to count the edge to parent.
                    # Actually, for undirected graphs, shortest cycle through 0 is:
                    # min over all edges (u,v) not in BFS tree of dist[u] + dist[v] + 1.
                    # This is O(E).
                    pass
        
        # Let's do a proper check for shortest cycle through 0
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        q = deque([0])
        
        while q:
            u = q.popleft()
            for v in ((u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n):
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        
        # Now check all edges for back-edges/cross-edges
        min_cycle = float('inf')
        for u in range(n):
            for v in ((u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n):
                if v > u: # Check each edge once
                    if parent[u] == v or parent[v] == u:
                        continue # Tree edge
                    # Cycle length is dist[u] + dist[v] + 1
                    # Note: This formula works if the LCA is 0? 
                    # Actually, for BFS from 0, dist[u] is distance from 0.
                    # The cycle is 0->...->u, edge u-v, v->...->0.
                    # Length = dist[u] + 1 + dist[v].
                    # This is the length of the cycle formed by the two paths from 0 and the edge.
                    # This is a valid cycle.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g_target:
                        return False
        
        if min_cycle < g_target:
            return False
        
        # If shortest cycle through 0 is >= g_target, we still need to check if there's a shorter cycle elsewhere.
        # For regular graphs, it's possible to have short cycles not involving 0.
        # We must check all vertices.
        # To save time, we only do this if the first check passes.
        # This full check is O(n * E) = O(n^2).
        
        # Optimization: If n is large, this is expensive.
        # But we only do it for candidates that pass the "through 0" test.
        # How many candidates pass?
        # For random a,b, the graph is likely to have small cycles through 0.
        
        # Full Girth Check
        # We can run BFS from every vertex up to depth g/2.
        # If we find a cycle < g, return False.
        
        # To make it faster, we can use the fact that we just need to find ANY cycle < g.
        # We can iterate u from 0 to n-1.
        # Run BFS from u, depth limit g/2.
        # If we encounter a node v with dist[v] < dist[u] (in terms of BFS order? No, standard BFS)
        # Actually, standard BFS from u finds shortest paths from u.
        # If we find an edge (x,y) where dist[x] + dist[y] + 1 < g, and the paths don't share nodes (except u)?
        # No, if we run BFS from u, and we find a back edge to a node w with dist[w] < dist[x] - 1,
        # we have a cycle.
        
        # Simpler: Just run BFS from each u, and track min cycle found.
        # To avoid O(n^2) cost for ALL n, we rely on the first filter.
        
        for start in range(n):
            dist_s = [-1] * n
            dist_s[start] = 0
            q_s = deque([start])
            min_c = float('inf')
            
            while q_s:
                u_s = q_s.popleft()
                d_u = dist_s[u_s]
                if d_u >= g_target // 2:
                    continue # No need to go deeper, cycle would be >= 2*(g/2)+1 > g
                for v_s in ((u_s + a) % n, (u_s - a) % n, (u_s + b) % n, (u_s - b) % n):
                    if dist_s[v_s] == -1:
                        dist_s[v_s] = d_u + 1
                        q_s.append(v_s)
                    else:
                        # Back edge or cross edge
                        # Cycle length = dist_s[u_s] + dist_s[v_s] + 1
                        # This is valid if the paths from start to u_s and start to v_s are disjoint?
                        # In BFS tree, they share the prefix from start to LCA.
                        # The cycle is formed by the tree path u_s -> LCA -> v_s and the edge u_s-v_s.
                        # Length = dist_s[u_s] + dist_s[v_s] - 2*dist_s[LCA] + 1.
                        # We don't know LCA easily.
                        # However, dist_s[u_s] + dist_s[v_s] + 1 is an UPPER bound on the cycle length?
                        # No, it's the length of the cycle that goes through start.
                        # If dist_s[u_s] + dist_s[v_s] + 1 < g_target, we definitely have a short cycle.
                        # If it's >= g_target, there might still be a shorter cycle not through start.
                        # But BFS from start finds the shortest path.
                        # Any cycle C has a vertex v. The shortest path from start to v is <= dist in C.
                        # This logic is tricky.
                        
                        # Standard Girth Check:
                        # A graph has girth >= g iff for all u, the number of vertices at distance i from u
                        # is at most k * (k-1)^(i-1) for i < g/2.
                        # This is a necessary condition. It is sufficient for bipartite graphs?
                        # No, but it's a very strong filter.
                        
                        # Let's just use the count check.
                        pass
            
            # Count check
            # For BFS from start, count nodes at each distance.
            # If count[d] > k * (k-1)^(d-1), there is a cycle of length <= 2d.
            # Specifically, if at distance d, we have more than max possible without cycles, we have a cycle.
            # Max nodes at distance d in a tree of degree k (root) is 1 + k + k(k-1) + ...
            # For k=4:
            # d=0: 1
            # d=1: 4
            # d=2: 12
            # d=3: 36
            # d=4: 108
            # d=5: 324
            # If g=10, we check up to d=4.
            # If count[4] > 108, there is a cycle of length <= 8?
            # Actually, if the graph is a tree up to depth d, the number of nodes is bounded.
            # If we exceed the bound, there is a cycle of length <= 2d.
            # So if we want girth >= 10, we need no cycles of length 4, 6, 8.
            # So we check up to d=4.
            # If count[d] > max_tree_nodes[d], then girth < 2d+2?
            # Let's use this as a fast filter.
            
            # We already did BFS for min_cycle_0. Let's reuse that logic for all nodes?
            # That's O(n^2).
            # For n=300, 300 BFS runs of size 300 is 90k operations.
            # 90k * 1000 candidates = 90M operations. Doable in 140s?
            # Python is slow. 10M ops/s is optimistic. 90M is ~10s.
            # But we have many n values.
            # We should only do the full check if the "through 0" check passes.
            # The "through 0" check is O(n).
            # So we do O(n) for all candidates, and O(n^2) for survivors.
            
        # Re-implementing the full check more carefully inside the loop
        # Actually, let's just return True if the "through 0" check passes, 
        # and assume that for random circulants, if there's a short cycle, it's likely through 0?
        # No, that's not safe.
        
        # Let's do the full check only if we are close to the deadline or if we find a candidate.
        # Actually, let's just do the full check for the first few n values.
        
        # To be safe and fast, let's use the "count" method for the full check.
        # It's O(n) per start node. Total O(n^2).
        
        # Let's optimize the loop.
        
        return True # Placeholder, need to implement full check properly

    # Correct Full Check Implementation
    def has_short_cycle(n, a, b, g_target):
        # Check if there is any cycle of length < g_target
        # Using BFS from each vertex, checking counts
        # Max nodes at depth d in k-regular tree (rooted):
        # d=0: 1
        # d=1: k
        # d=2: k(k-1)
        # d=3: k(k-1)^2
        # ...
        # d: k(k-1)^(d-1) for d>=1
        
        # If count[d] > k * (k-1)^(d-1), then there is a cycle of length <= 2d.
        # We want girth >= g_target.
        # So we need no cycles of length < g_target.
        # So we need no cycles of length 4, 6, ..., g_target-2.
        # So we check depths d such that 2d < g_target.
        # i.e., d < g_target / 2.
        
        max_depth = (g_target - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            counts = [0] * (max_depth + 1)
            counts[0] = 1
            
            while q:
                u = q.popleft()
                d = dist[u]
                if d >= max_depth:
                    continue
                for v in ((u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n):
                    if dist[v] == -1:
                        dist[v] = d + 1
                        q.append(v)
                        if d + 1 <= max_depth:
                            counts[d + 1] += 1
                    else:
                        # If v is already visited, we have a cycle.
                        # The length of the cycle is dist[u] + dist[v] + 1 - 2*dist[lca]
                        # We don't know lca.
                        # But if dist[u] + dist[v] + 1 < g_target, it's a short cycle?
                        # Not necessarily, if lca is deep.
                        # However, the count check is more robust for "existence".
                        pass
            
            # Check counts
            for d in range(1, max_depth + 1):
                max_allowed = 4 * (3 ** (d - 1))
                if counts[d] > max_allowed:
                    return True # Has short cycle
        
        return False # No short cycle found

    # Search
    for n in range(243, 321):
        if time.time() > deadline:
            break
            
        # Symmetry: a < b, a < n/2, b < n/2
        # Also a != b, a != n-b
        # We can restrict a to 1..n//2, b to a+1..n//2
        # But we must ensure steps are distinct.
        
        found = False
        for a in range(1, n // 2 + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, n // 2 + 1):
                # Check distinctness
                if a == b or a == n - b:
                    continue
                # Also check if 2a == 0 or 2b == 0 (loops/multi-edges)
                # a != n-a => 2a != n
                # b != n-b => 2b != n
                if 2 * a == n or 2 * b == n:
                    continue
                if a == 0 or b == 0:
                    continue
                
                # Fast check: shortest cycle through 0
                # We can just use the has_short_cycle function, but it's O(n^2).
                # Let's do a quick O(n) check for cycle through 0 first.
                
                # Quick check:
                dist = [-1] * n
                dist[0] = 0
                q = deque([0])
                min_c = float('inf')
                while q:
                    u = q.popleft()
                    du = dist[u]
                    if du >= g // 2:
                        continue
                    for v in ((u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n):
                        if dist[v] == -1:
                            dist[v] = du + 1
                            q.append(v)
                        else:
                            # Back edge
                            # Cycle length approx dist[u] + dist[v] + 1
                            # This is the cycle through 0.
                            if dist[u] + dist[v] + 1 < min_c:
                                min_c = dist[u] + dist[v] + 1
                            if min_c < g:
                                break
                    if min_c < g:
                        break
                
                if min_c < g:
                    continue
                
                # Passed quick check. Do full check.
                if not has_short_cycle(n, a, b, g):
                    # Valid!
                    edges = []
                    for i in range(n):
                        for s in (a, n - a, b, n - b):
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return luw_construct(k, g)
