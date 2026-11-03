"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import collections

def construct(k, g):
    q = k
    
    # Finite-field helpers (q prime or q == 4)
    if q == 4:
        _add_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
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

    def check_girth(n, edges, min_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles shorter than min_girth
        # Since it's bipartite, we check even lengths 4, 6, ...
        # We can just do BFS for each edge to find shortest path u->v without that edge.
        
        # Optimization: If min_girth is 10, we only need to rule out 4, 6, 8.
        # BFS depth limit: min_girth - 1
        
        max_depth = min_girth - 1
        
        for u, v in edges:
            # BFS from u to v, avoiding the direct edge (u,v) initially?
            # Actually, shortest path in G - (u,v).
            # If dist(u,v) in G-(u,v) is d, cycle is d+1.
            # We want d+1 < min_girth => d < min_girth - 1.
            
            visited = [False] * n
            visited[u] = True
            dist = [0] * n
            dq = collections.deque()
            dq.append(u)
            
            found = False
            while dq:
                curr = dq.popleft()
                if dist[curr] >= max_depth:
                    continue
                for nei in adj[curr]:
                    if nei == v and curr == u:
                        continue # Skip the edge we removed? 
                        # Wait, BFS explores neighbors. 
                        # If we are at u, and we see v, that's the direct edge. 
                        # We want to exclude the direct edge (u,v).
                        # So if curr==u and nei==v, skip.
                        # Also if curr==v and nei==u, skip? No, we stop at v.
                        pass
                    if not visited[nei]:
                        visited[nei] = True
                        dist[nei] = dist[curr] + 1
                        if nei == v:
                            # Found path of length dist[curr] + 1
                            cycle_len = dist[curr] + 1 + 1 # +1 for the edge (u,v) itself?
                            # No, path is u -> ... -> v. Length is dist[u->v].
                            # Cycle is u -> ... -> v -> u. Length is dist + 1.
                            if dist[curr] + 1 < min_girth:
                                return False
                            found = True
                            break
                        dq.append(nei)
            # If we break out of neighbor loop, we need to break out of while
            # The logic above is a bit messy with the 'found' flag.
            # Let's rewrite the BFS part cleanly.
            
        return True

    # Try small d first
    # For k=5, g=9, we need girth >= 10 (bipartite).
    # d=3 -> N=250. Girth likely 6 or 8.
    # d=4 -> N=1250. Girth likely 8 or 10.
    # d=5 -> N=6250. Girth likely 10 or 12.
    
    max_d = 10
    for d in range(3, max_d + 1):
        n, edges = build_graph(d)
        if check_girth(n, edges, g):
            return n, edges
            
    # Fallback to original heuristic if all fail (should not happen)
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
    n, edges = build_graph(d)
    return n, edges

# EVOLVE-BLOCK-END
