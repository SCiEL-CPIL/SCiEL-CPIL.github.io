"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import collections

def construct(k, g):
    # Deadline guard: evaluator kills at 150s; we aim to finish well before.
    deadline = time.time() + 130.0
    
    q = k  # For k=5, q=5 (prime)
    
    # Finite-field helpers
    if q == 4:
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q

    def get_rules(d):
        rules = {}
        rules[2] = (1, 1)
        rules[3] = (2, 1)
        rules[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            base = c - r
            candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
            rules[c] = candidates[r]
        return rules

    def build_graph(d):
        """
        Build the LUW graph for parameter d.
        Returns (n, edges) for the connected component of the zero point.
        """
        rules = get_rules(d)
        start_key = (0, (0,) * (d + 1))
        label = {start_key: 0}
        queue = [start_key]
        edge_list = []
        
        for vkey in queue:
            side, coords = vkey
            vidx = label[vkey]
            for first_val in range(q):
                ncoords = [0] * (d + 1)
                ncoords[1] = first_val
                for c in range(2, d + 1):
                    a, b = rules[c]
                    if side == 0:
                        ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                    else:
                        ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
                nkey = (1 - side, tuple(ncoords))
                if nkey not in label:
                    label[nkey] = len(queue)
                    queue.append(nkey)
                if side == 0:
                    edge_list.append((vidx, label[nkey]))
        return len(queue), edge_list

    def check_girth_fast(n, edges, min_girth):
        """
        Check if the graph has girth >= min_girth.
        For bipartite graphs, we only need to check even cycles 4, 6, ... < min_girth.
        We use BFS from each vertex to find the shortest cycle through it.
        Actually, a more efficient method: for each edge (u,v), check if there's a path
        from u to v of length < min_girth - 1 in G - {(u,v)}.
        
        Given the time constraint and that we expect d=4 to work for k=5, g=9,
        we do a quick check. If it's too slow, we assume it works for the LUW family
        when d is chosen correctly.
        
        However, to be safe, let's do a proper check but optimized.
        Since n is at most ~1250 for d=4, q=5, this is feasible.
        """
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For bipartite graphs, shortest cycle is even.
        # We check for cycles of length 4, 6, ..., min_girth-1 (if min_girth is odd, we check up to min_girth-1 which is even? No, girth is the length of the shortest cycle. If girth >= 9, there are no cycles of length 4, 6, 8. So we check 4, 6, 8.)
        
        # A cycle of length L exists if for some edge (u,v), there is a path from u to v of length L-1 in G - {(u,v)}.
        # Since the graph is bipartite, L must be even.
        
        max_cycle_len = min_girth - 1
        if max_cycle_len % 2 == 1:
            max_cycle_len -= 1  # Only even cycles
        
        # We will check for each edge if there is a path of length < max_cycle_len (i.e., <= max_cycle_len - 1)
        # Actually, we want to ensure NO cycle of length < min_girth.
        # So we check if there is a cycle of length 4, 6, ..., min_girth-1 (if min_girth is odd, up to min_girth-1 is even? No, if min_girth=9, we check 4, 6, 8).
        
        # Let's just do a BFS from each vertex u, and for each neighbor v, check if the shortest path from u to v (excluding the direct edge) is < min_girth - 1.
        # This is O(n * m) which might be too slow if n is large, but for n~1250, m~3125, it's ~4 million ops, which is fine.
        
        # Optimization: We can just check for 4-cycles first, as they are the most common.
        # If no 4-cycles, check 6-cycles, etc.
        
        # Let's use the standard method: for each edge (u,v), remove it and find dist(u,v). If dist < min_girth - 1, return False.
        # To speed up, we can use BFS with a depth limit.
        
        # But doing this for every edge is expensive.
        # Alternative: For each vertex u, do a BFS up to depth (min_girth-1)/2. If we encounter a neighbor v of u at depth d, and d+1 < min_girth, we have a cycle.
        # Actually, if we are at vertex x at depth d from u, and x is a neighbor of u, then we have a cycle of length d+1.
        # We need d+1 < min_girth => d < min_girth - 1.
        
        # Let's implement this.
        limit = (min_girth - 1) // 2  # Max depth to check from u
        
        for u in range(n):
            visited = [-1] * n
            visited[u] = 0
            dq = collections.deque([u])
            while dq:
                curr = dq.popleft()
                if visited[curr] >= limit:
                    continue
                for nei in adj[curr]:
                    if nei == u:
                        # Found a cycle of length visited[curr] + 1
                        if visited[curr] + 1 < min_girth:
                            return False
                    elif visited[nei] == -1:
                        visited[nei] = visited[curr] + 1
                        dq.append(nei)
        
        return True

    # Determine the smallest d such that the LUW graph has girth >= g.
    # For odd d, girth >= d + 5.
    # For even d, girth >= d + 4.
    
    # We want the smallest n, which corresponds to the smallest d.
    # So we try d = 3, 4, 5, ... until we find one that works.
    
    best_n = float('inf')
    best_edges = []
    
    # We expect d=4 to work for k=5, g=9 (girth >= 8? No, girth >= 4+4=8. We need 9. So d=4 might not be enough if the bound is tight. But the bound is a lower bound on the girth. The actual girth might be higher.)
    # Actually, for k=5, d=4, the LUW graph has girth 10 (I think). Let's check.
    # If d=4, girth >= 8. It might be 8 or 10. If it's 8, it fails for g=9.
    # If d=5, girth >= 10. This should work.
    
    # Let's try d starting from 3.
    for d in range(3, 11):
        if time.time() > deadline - 5:
            break
        n, edges = build_graph(d)
        if n < best_n:
            # Check girth
            if check_girth_fast(n, edges, g):
                best_n = n
                best_edges = edges
                # Since we are increasing d, n increases exponentially.
                # The first d that works gives the smallest n.
                # So we can break here if we are confident.
                # But let's check if a smaller d was already tried.
                # We started from d=3, so this is the smallest d that works.
                break
        
    if best_n != float('inf'):
        return best_n, best_edges
    
    # Fallback: use the largest d we tried
    d = 10
    n, edges = build_graph(d)
    return n, edges

# EVOLVE-BLOCK-END
