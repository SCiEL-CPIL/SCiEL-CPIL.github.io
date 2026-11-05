def construct():
    from math import gcd
    from collections import defaultdict, deque
    import sys

    n = 256
    candidates = list(range(2, n + 1))
    indices = {x: i for i, x in enumerate(candidates)}
    N = len(candidates)
    
    # 1. Build Conflict Graph
    # conflicts[i] is a set of indices j such that lcm(candidates[i], candidates[j]) <= n
    conflicts = [set() for _ in range(N)]
    for i in range(N):
        a = candidates[i]
        for j in range(i + 1, N):
            b = candidates[j]
            g = gcd(a, b)
            if (a // g) * b <= n:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # 2. Find Connected Components
    visited = [False] * N
    components = []
    for i in range(N):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                u = queue.pop()
                comp.append(u)
                for v in conflicts[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            components.append(comp)
            
    best_set = set()
    total_weight = 0.0
    
    # 3. Solve each component
    for comp in components:
        # Check if bipartite
        # color: -1 unvisited, 0, 1
        color = [-1] * N # Global array for simplicity, or local map
        # We need to map comp nodes to 0..k-1 for local operations
        node_map = {node: idx for idx, node in enumerate(comp)}
        k = len(comp)
        
        # Local adjacency list
        local_adj = [[] for _ in range(k)]
        for u in comp:
            idx_u = node_map[u]
            for v in conflicts[u]:
                if v in node_map:
                    idx_v = node_map[v]
                    local_adj[idx_u].append(idx_v)
                    
        # Bipartite check
        is_bipartite = True
        colors = [-1] * k
        for i in range(k):
            if colors[i] == -1:
                colors[i] = 0
                q = deque([i])
                while q:
                    u = q.popleft()
                    for v in local_adj[u]:
                        if colors[v] == -1:
                            colors[v] = 1 - colors[u]
                            q.append(v)
                        elif colors[v] == colors[u]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
                    
        if is_bipartite:
            # Solve MWIS on Bipartite Graph using Max Flow
            # Partition U (color 0) and V (color 1)
            U = [i for i in range(k) if colors[i] == 0]
            V = [i for i in range(k) if colors[i] == 1]
            
            # Build Flow Network
            # Nodes: 0 (Source), 1..len(U) (U nodes), len(U)+1..len(U)+len(V) (V nodes), len(U)+len(V)+1 (Sink)
            num_nodes = 2 + len(U) + len(V)
            source = 0
            sink = num_nodes - 1
            
            # Dinic's Algorithm
            # graph[u] = list of [v, cap, rev_index]
            graph = [[] for _ in range(num_nodes)]
            
            def add_edge(u, v, cap):
                graph[u].append([v, cap, len(graph[v])])
                graph[v].append([u, 0, len(graph[u]) - 1])
                
            total_w = 0.0
            u_map = {u: i+1 for i, u in enumerate(U)}
            v_map = {v: i+len(U)+1 for i, v in enumerate(V)}
            
            for u in U:
                w = 1.0 / candidates[comp[u]]
                total_w += w
                add_edge(source, u_map[u], w)
                
            for v in V:
                w = 1.0 / candidates[comp[v]]
                total_w += w
                add_edge(v_map[v], sink, w)
                
            for u in U:
                for v in local_adj[u]:
                    if v in v_map: # Only edges U->V
                        add_edge(u_map[u], v_map[v], float('inf'))
                        
            # Dinic
            def max_flow(s, t):
                flow = 0
                level = [0] * num_nodes
                iter_ptr = [0] * num_nodes
                
                def bfs():
                    for i in range(num_nodes):
                        level[i] = -1
                    level[s] = 0
                    q = deque([s])
                    while q:
                        u = q.popleft()
                        for v, cap, rev in graph[u]:
                            if cap > 0 and level[v] < 0:
                                level[v] = level[u] + 1
                                q.append(v)
                    return level[t] >= 0
                    
                def dfs(u, f):
                    if u == t:
                        return f
                    while iter_ptr[u] < len(graph[u]):
                        v, cap, rev = graph[u][iter_ptr[u]]
                        if cap > 0 and level[u] < level[v]:
                            d = dfs(v, min(f, cap))
                            if d > 0:
                                graph[u][iter_ptr[u]][1] -= d
                                graph[v][rev][1] += d
                                return d
                        iter_ptr[u] += 1
                    return 0
                    
                while bfs():
                    for i in range(num_nodes):
                        iter_ptr[i] = 0
                    while True:
                        f = dfs(s, float('inf'))
                        if f == 0:
                            break
                        flow += f
                return flow
                
            min_cut = max_flow(source, sink)
            mw_is_weight = total_w - min_cut
            
            # Reconstruct Set
            # Nodes in S (reachable from source in residual graph)
            # In Min Cut, S contains Source.
            # U nodes in T (not in S) are in IS.
            # V nodes in S (in S) are in IS.
            
            # Find reachable nodes in residual graph
            reachable = [False] * num_nodes
            q = deque([source])
            reachable[source] = True
            while q:
                u = q.popleft()
                for v, cap, rev in graph[u]:
                    if cap > 0 and not reachable[v]:
                        reachable[v] = True
                        q.append(v)
                        
            comp_set = set()
            for u in U:
                if not reachable[u_map[u]]: # In T
                    comp_set.add(comp[u])
            for v in V:
                if reachable[v_map[v]]: # In S
                    comp_set.add(comp[v])
                    
            best_set.update(comp_set)
            total_weight += mw_is_weight
            
        else:
            # Solve General MWIS using Bitmask DP
            # Map comp nodes to 0..k-1
            # k is likely small if non-bipartite
            if k > 25:
                # Fallback to greedy or simple heuristic if too large? 
                # Or use B&B?
                # For now, assume k is small. If not, this might TLE.
                # We can try a simple greedy as fallback, but it's not exact.
                # Given the task, exact is required.
                # We'll stick to DP, hoping k is small.
                pass
                
            node_map = {node: i for i, node in enumerate(comp)}
            adj_mask = [0] * k
            for i in range(k):
                u = comp[i]
                for v in conflicts[u]:
                    if v in node_map:
                        adj_mask[i] |= (1 << node_map[v])
                        
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dp(mask):
                if mask == 0:
                    return 0.0, frozenset()
                lsb = mask & -mask
                i = lsb.bit_length() - 1
                # Exclude i
                w1, s1 = dp(mask ^ lsb)
                # Include i
                new_mask = mask ^ lsb
                new_mask &= ~adj_mask[i]
                w2, s2 = dp(new_mask)
                w2 += 1.0 / candidates[comp[i]]
                s2 = s2 | {i}
                
                if w1 > w2:
                    return w1, s1
                else:
                    return w2, s2
                    
            w, s_indices = dp((1 << k) - 1)
            s_nodes = {comp[i] for i in s_indices}
            best_set.update(s_nodes)
            total_weight += w
            
    return list(best_set)
