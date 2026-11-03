# EVOLVE-BLOCK-START
def construct(k, g):
    q = k
    
    def f_add(a, b): return (a + b) % q
    def f_sub(a, b): return (a - b) % q
    def f_mul(a, b): return (a * b) % q

    def build_luw(d, rules):
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
        from collections import deque
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each vertex to find shortest cycle
        # We only care if girth < min_girth
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q_bfs = deque([start])
            while q_bfs:
                u = q_bfs.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q_bfs.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Try d=4 with various rules
    # Standard LUW rules for d=4:
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2) or (2,1) etc.
    
    base_rules = {2: (1, 1), 3: (2, 1)}
    
    candidates = []
    # Try a few variations for c=4
    for r4 in [(1, 2), (2, 1), (1, 1), (2, 2)]:
        rules = base_rules.copy()
        rules[4] = r4
        candidates.append(rules)

    for rules in candidates:
        n, edges = build_luw(4, rules)
        if check_girth(n, edges, g):
            return n, edges

    # Fallback to d=5
    d = 5
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates_r = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates_r[r]
        
    n, edges = build_luw(d, rules)
    return n, edges
# EVOLVE-BLOCK-END
