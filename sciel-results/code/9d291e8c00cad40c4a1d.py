<<<<<<< SEARCH
def _bfs_short_cycle_nb(src, adj, n, g):
    """
    Non-backtracking BFS from src. Returns True if a cycle shorter than g is found.
    State: (current, previous). We track the minimum depth for each state.
    """
    # State: (u, prev) where prev is the vertex we came from (or -1 for start)
    dist = {}
    queue = deque()
    start_state = (src, -1)
    dist[start_state] = 0
    queue.append(start_state)
    
    max_depth = g - 1
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_depth:
            continue
        for v in adj[u]:
            if v == prev:
                continue
            ns = (v, u)
            nd = d + 1
            if nd > max_depth:
                continue
            if ns not in dist:
                dist[ns] = nd
                if v == src and u != -1:
                    # Found a cycle of length nd
                    if nd < g:
                        return True
                queue.append(ns)
            # If ns was already visited at a smaller or equal depth, skip
    return False


def _multi_base_voltage_lift(k, g, deadline):
    """
    Voltage lift over multiple base graphs (Petersen, Heawood, etc.)
    with randomized voltage assignments.
    """
    if k != 3:
        return None
    
    # Base graphs to try
    base_graphs = [
        # Petersen (10 vertices)
        {
            'n': 10,
            'edges': [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,9),(9,5),(0,5),(1,6),(2,7),(3,8),(4,9)]
        },
        # 3-regular graph on 12 vertices (cubical)
        {
            'n': 12,
            'edges': [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(8,9),(9,10),(10,11),(11,8),(0,4),(1,5),(2,6),(3,7)]
        },
    ]
    
    best_result = None
    best_n = float('inf')
    
    for bg in base_graphs:
        bn = bg['n']
        b_edges = bg['edges']
        
        # Build base adjacency
        b_adj = [[] for _ in range(bn)]
        for u, v in b_edges:
            b_adj[u].append(v)
            b_adj[v].append(u)
        
        # Try different m values
        m_min = max(1, (510 // bn) + 1)
        m_max = min(300, m_min + 50)
        
        for m in range(m_min, m_max + 1):
            if time.time() > deadline - 10:
                return best_result
            
            n_lift = bn * m
            if n_lift >= best_n:
                break
            
            # Try several random voltage assignments
            for trial in range(20):
                if time.time() > deadline - 5:
                    return best_result
                
                voltages = {}
                for u, v in b_edges:
                    vol = random.randint(0, m - 1)
                    voltages[(u, v)] = vol
                    voltages[(v, u)] = (-vol) % m
                
                if _lift_girth_check_fast(b_adj, voltages, m, g):
                    # Build the graph
                    edges = []
                    for u, v in b_edges:
                        vol = voltages[(u, v)]
                        for i in range(m):
                            a = u * m + i
                            b = v * m + ((i + vol) % m)
                            if a < b:
                                edges.append((a, b))
                            else:
                                edges.append((b, a))
                    
                    if len(edges) == bn * m * k // 2:
                        if n_lift < best_n:
                            best_n = n_lift
                            best_result = (n_lift, edges)
                        break
    
    return best_result


def _lift_girth_check_fast(base_adj, voltages, m, g):
    """
    Fast girth check for voltage lift.
    A cycle of length L in the lift corresponds to a closed non-backtracking walk
    of length L in the base with total voltage 0 mod m.
    We check all closed non-backtracking walks of length < g.
    """
    bn = len(base_adj)
    
    for start in range(bn):
        # State: (current, prev, voltage_sum)
        # BFS by depth
        visited = set()
        queue = deque()
        start_state = (start, -1, 0)
        visited.add(start_state)
        queue.append((start, -1, 0, 0))
        
        while queue:
            u, prev, vol, depth = queue.popleft()
            if depth >= g - 1:
                continue
            
            for v in base_adj[u]:
                if v == prev:
                    continue
                new_vol = (vol + voltages.get((u, v), 0)) % m
                new_state = (v, u, new_vol)
                if new_state in visited:
                    continue
                visited.add(new_state)
                new_depth = depth + 1
                
                if v == start and new_vol == 0 and new_depth >= 2:
                    return False  # Found short cycle
                
                if new_depth < g - 1:
                    queue.append((v, u, new_vol, new_depth))
    
    return True
=======
def _bfs_short_cycle_nb(src, adj, n, g):
    """
    Non-backtracking BFS from src. Returns True if a cycle shorter than g is found.
    """
    dist = {}
    queue = deque()
    start_state = (src, -1)
    dist[start_state] = 0
    queue.append(start_state)
    
    max_depth = g - 1
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_depth:
            continue
        for v in adj[u]:
            if v == prev:
                continue
            ns = (v, u)
            nd = d + 1
            if nd > max_depth:
                continue
            if ns not in dist:
                dist[ns] = nd
                if v == src and u != -1:
                    if nd < g:
                        return True
                queue.append(ns)
    return False
>>>>>>> REPLACE

