import time
import random
import sys

def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar construction
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

def _generate_random_regular(n, k):
    """Generate a random simple k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    attempts = 0
    max_attempts = 500
    while attempts < max_attempts:
        attempts += 1
        random.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            return adj
    return None

def _check_girth(adj, g):
    """Check if graph has girth >= g. Returns True if girth >= g, else False.
    Uses BFS from each vertex up to depth (g-2)//2."""
    n = len(adj)
    max_depth = (g - 2) // 2
    for start in range(n):
        # BFS from start, tracking parent to avoid backtracking
        visited = {}  # node -> (distance, parent)
        queue = [(start, -1, 0)]
        while queue:
            u, parent, dist = queue.pop(0)
            if u in visited:
                if dist < visited[u][0]:
                    continue
                # If we've seen u before with a different parent, we found a cycle
                # Actually, we need to be more careful. Let's use a different approach.
                continue
            visited[u] = (dist, parent)
            if dist >= max_depth:
                continue
            for v in adj[u]:
                if v == parent:
                    continue
                if v not in visited:
                    queue.append((v, u, dist + 1))
                else:
                    # Found a cycle. Its length is dist + 1 + visited[v][0]
                    cycle_len = dist + 1 + visited[v][0]
                    if cycle_len < g:
                        return False
    return True

def _find_short_cycle_vertices(adj, g):
    """Find a short cycle (length < g) and return its vertices as a set.
    Returns None if no short cycle found."""
    n = len(adj)
    max_depth = (g - 2) // 2
    for start in range(n):
        # BFS from start
        dist = {}
        parent = {}
        dist[start] = 0
        parent[start] = -1
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle through u-v
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct the cycle
                        # Path from start to u, edge u-v, path from start to v
                        path_u = []
                        cur = u
                        while cur != -1:
                            path_u.append(cur)
                            cur = parent[cur]
                        path_v = []
                        cur = v
                        while cur != -1:
                            path_v.append(cur)
                            cur = parent[cur]
                        # The cycle is the union of these paths (minus the common prefix)
                        # Actually, for a simple cycle, we need the symmetric difference
                        # Let's just return the set of vertices on the shorter path
                        cycle_verts = set(path_u) | set(path_v)
                        return cycle_verts
    return None

def _try_switching(n, k, g, time_limit=100.0):
    start_time = time.time()
    
    # 1. Generate initial graph
    adj = _generate_random_regular(n, k)
    if adj is None:
        return None
    
    # Check if girth is already >= g
    if _check_girth(adj, g):
        edge_list = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edge_list.append((i, j))
        return n, edge_list
    
    # 2. Switching loop
    while time.time() - start_time < time_limit:
        # Find a short cycle
        cycle_verts = _find_short_cycle_vertices(adj, g)
        if cycle_verts is None:
            edge_list = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edge_list.append((i, j))
            return n, edge_list
        
        # Try to break the cycle by switching
        # Pick an edge (u,v) on the cycle, and find x,y to switch
        success = False
        for _ in range(100):
            # Pick a random edge within the cycle vertices
            cycle_list = list(cycle_verts)
            if len(cycle_list) < 2:
                break
            u = random.choice(cycle_list)
            # Pick a neighbor v of u that's also in the cycle
            v_candidates = [v for v in adj[u] if v in cycle_verts]
            if not v_candidates:
                continue
            v = random.choice(v_candidates)
            
            # Pick x randomly (not in cycle), y a neighbor of x
            x = random.randint(0, n - 1)
            if x in cycle_verts:
                continue
            if not adj[x]:
                continue
            y = random.choice(list(adj[x]))
            if y in cycle_verts or y == u or y == v:
                continue
            if x in adj[u] or y in adj[v]:
                continue
            
            # Perform switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            success = True
            break
        
        if not success:
            # Try a random switch
            u = random.randint(0, n - 1)
            if not adj[u]:
                continue
            v = random.choice(list(adj[u]))
            x = random.randint(0, n - 1)
            if x == u or x == v or x not in adj:
                continue
            if not adj[x]:
                continue
            y = random.choice(list(adj[x]))
            if y == u or y == v or y == x:
                continue
            if x in adj[v] or y in adj[u]:
                continue
            
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
    
    return None

def construct(k, g):
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    if k == 5 and g == 9:
        # Try switching on smaller n first
        target_ns = [1116, 1120, 1150]
        time_per_attempt = 40.0
        
        for n in target_ns:
            if time.time() > deadline - 20.0:
                break
            res = _try_switching(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
    
    # Fallback to LUW
    return _luw_construct(k, g)
