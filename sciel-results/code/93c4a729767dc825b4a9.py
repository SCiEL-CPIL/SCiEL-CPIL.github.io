import time
import sys
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # Moore bound: n >= k + k*(k-1)^(ceil((g-1)/2)/2 - 1) * (something)
    # For k-regular graph of girth g, Moore bound M(k,g):
    # If g is even: M = k * ((k-1)^(g/2 - 1) - 1) / (k - 2)
    # If g is odd: M = 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i = 1 + k * ((k-1)^((g-1)/2) - 1)/(k-2)
    
    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2 - 1
            return k * ((k-1)**h - 1) // (k - 2) + 1
        else:
            h = (g - 1) // 2
            return 1 + k * ((k-1)**h - 1) // (k - 2)
    
    mb = moore_bound(k, g)
    
    # LUW construction
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
                
    luw_n = len(order)
    luw_edges = edges
    
    # Now try to improve via edge swaps
    # Build adjacency list
    n = luw_n
    adj = [[] for _ in range(n)]
    for u, v in luw_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify girth with non-backtracking walks
    def check_girth(n, adj, g):
        """Check that no cycle shorter than g exists using non-backtracking walks."""
        for s in range(n):
            # BFS on non-backtracking walks from s
            # State: (current_vertex, previous_vertex)
            # We want to find if there's a non-backtracking walk of length < g that returns to s
            # Actually, we need to check: is there a cycle of length < g?
            # A cycle of length L corresponds to a non-backtracking closed walk of length L.
            # Use BFS from each directed edge (s, t) to see if we can return to s in < g steps
            # without backtracking.
            
            # More efficient: BFS from s, tracking the previous vertex to avoid backtracking
            # dist[(v, u)] = shortest non-backtracking walk length from s to v that came from u
            # We want to know if there's a path of length < g from s back to s.
            # Equivalently: is there a vertex v at distance < g-1 from s (via non-backtracking)
            # such that v is adjacent to s and v != parent of s in the walk?
            
            # Simpler approach: for each starting directed edge (s, v0), do BFS
            # tracking (current, prev) and check if we reach (s, u) where u != v0
            # and path length < g.
            
            # Actually the standard way: 
            # For each vertex s, do a BFS where state is (v, parent).
            # Start from (s, -1) with distance 0.
            # From (v, parent), we can go to (w, v) for all w in adj[v] where w != parent.
            # If we reach (s, w) with distance < g, then we have a cycle of length < g.
            # But we need to be careful: the walk must be a valid cycle.
            
            # Let's do it properly:
            # dist[(v, p)] = min length of non-backtracking walk from s to v with previous vertex p
            # Initialize: for each v in adj[s], dist[(v, s)] = 1
            # Then BFS. If we find dist[(s, v)] < g for some v in adj[s], we have a cycle.
            # But wait, we start at s and need to return to s. The walk is s -> v1 -> ... -> vL -> s
            # where vL is adjacent to s. The non-backtracking condition means v_{i+1} != v_{i-1}.
            # So the state when we're at s coming from vL is (s, vL).
            # We need the total length L < g.
            
            # Let's track: from s, go to each neighbor v (length 1, state (v, s))
            # Then continue. If we reach state (s, v) with total length L < g, cycle found.
            
            from collections import deque
            q2 = deque()
            dist = {}
            for v in adj[s]:
                q2.append((v, s))
                dist[(v, s)] = 1
            
            while q2:
                cur, prev = q2.popleft()
                d_cur = dist[(cur, prev)]
                if d_cur >= g - 1:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    if nxt == s:
                        # Found a cycle of length d_cur + 1
                        if d_cur + 1 < g:
                            return False
                    else:
                        if (nxt, cur) not in dist:
                            dist[(nxt, cur)] = d_cur + 1
                            q2.append((nxt, cur))
        return True
    
    # Verify the LUW graph
    girth_ok = check_girth(n, adj, g)
    
    # Try edge swaps to reduce vertex count is not directly possible.
    # Instead, try to find a smaller graph.
    
    # For k=4, g=10, the known best is 320. LUW gives 1024.
    # Let's try a random construction approach for small cases.
    
    # Actually, let's try the (k,g) specific known constructions.
    # For k=4, g=10: The Robertson graph has 192 vertices but girth 8.
    # The Foster census... 
    
    # Let's try a different approach: use the known fact that for k=4, g=10,
    # there's a construction based on the (4,10)-cage with 320 vertices.
    # Without the exact edge list, we can't use it directly.
    
    # Try random graph construction with girth verification
    def try_random_construction(k, g, max_n, max_time):
        """Try to construct a k-regular graph of girth g with at most max_n vertices."""
        target_n = max_n
        # Try various n values from moore bound upward
        for n_try in range(mb, target_n + 1):
            if time.time() > max_time:
                break
            # Try to build a k-regular graph on n_try vertices
            # Use configuration model + girth check
            for attempt in range(50):
                if time.time() > max_time:
                    break
                # Configuration model
                half_edges = []
                for v in range(n_try):
                    for _ in range(k):
                        half_edges.append(v)
                random.shuffle(half_edges)
                edges = set()
                valid = True
                for i in range(0, len(half_edges), 2):
                    u, v = half_edges[i], half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    if (min(u,v), max(u,v)) in edges:
                        valid = False
                        break
                    edges.add((min(u,v), max(u,v)))
                if not valid:
                    continue
                edge_list = list(edges)
                if len(edge_list) != n_try * k // 2:
                    continue
                # Build adj
                adj2 = [[] for _ in range(n_try)]
                for u, v in edge_list:
                    adj2[u].append(v)
                    adj2[v].append(u)
                # Check regular
                if any(len(adj2[v]) != k for v in range(n_try)):
                    continue
                # Check girth
                if check_girth(n_try, adj2, g):
                    return n_try, edge_list
        return None
    
    # Only try random if LUW is very large
    if luw_n > mb * 2 and g <= 12:
        result = try_random_construction(k, g, min(luw_n - 1, mb + 100), start_time + 100)
        if result is not None:
            return result
    
    return luw_n, luw_edges
# EVOLVE-BLOCK-END
