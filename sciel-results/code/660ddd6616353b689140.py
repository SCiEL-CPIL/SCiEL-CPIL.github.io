def construct():
    N = 256
    S = set()
    
    def solve_vertex_cover(nodes, edges, weights):
        if not edges:
            return 0.0, set()
            
        adj = {}
        for n in nodes:
            adj[n] = []
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        visited = set()
        total_cost = 0.0
        total_nodes = set()
        
        for start in nodes:
            if start in visited:
                continue
                
            component_nodes = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component_nodes.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
                        
            comp_edges = [(u, v) for u, v in edges if u in visited and v in visited]
            
            deg = {n: 0 for n in component_nodes}
            for u, v in comp_edges:
                deg[u] += 1
                deg[v] += 1
                
            is_cycle = all(d == 2 for d in deg.values())
            
            def dp_path(order):
                # Returns (cost, set_of_nodes)
                if len(order) == 1:
                    return 0.0, set()
                    
                # dp[i][0] = (cost, set) for path ending at i, i not in cover
                # dp[i][1] = (cost, set) for path ending at i, i in cover
                
                # We need to reconstruct the set, so we store the choice.
                # To save memory, we can just store the cost and then backtrack.
                # But the path is short. Let's store the sets directly.
                
                # dp0: list of (cost, set)
                # dp1: list of (cost, set)
                
                dp0 = [(0.0, set())] # index 0, node not in cover
                dp1 = [(weights[order[0]], {order[0]})] # index 0, node in cover
                
                for i in range(1, len(order)):
                    w = weights[order[i]]
                    node = order[i]
                    
                    # New dp0[i]: node not in cover. Previous node MUST be in cover.
                    # So take dp1[i-1]
                    c0, s0 = dp1[i-1]
                    new_dp0 = (c0, set(s0))
                    
                    # New dp1[i]: node in cover. Previous node can be 0 or 1.
                    c0_prev, s0_prev = dp0[i-1]
                    c1_prev, s1_prev = dp1[i-1]
                    
                    if c0_prev <= c1_prev:
                        new_dp1 = (c0_prev + w, set(s0_prev) | {node})
                    else:
                        new_dp1 = (c1_prev + w, set(s1_prev) | {node})
                        
                    dp0.append(new_dp0)
                    dp1.append(new_dp1)
                    
                # End of path
                best0 = dp0[-1]
                best1 = dp1[-1]
                
                if best0[0] <= best1[0]:
                    return best0
                else:
                    return best1
                    
            if is_cycle:
                # Find order
                v1 = component_nodes[0]
                order = [v1]
                prev = None
                curr = v1
                while True:
                    nexts = [n for n in adj[curr] if n != prev]
                    if not nexts:
                        break
                    if prev is None:
                        next_node = nexts[0]
                    else:
                        if nexts[0] == v1 and len(order) == len(component_nodes):
                            break
                        next_node = nexts[0]
                    order.append(next_node)
                    prev = curr
                    curr = next_node
                    
                # Case 1: v1 in cover
                # Force v1 in cover in dp
                # We can modify dp_path to accept constraints, or just run it and filter?
                # Easier: Run dp_path, but we need to force v1.
                # Let's create a modified dp that takes fix_start and fix_end.
                
                def dp_path_constrained(order, fix_start=None, fix_end=None):
                    if len(order) == 1:
                        if fix_start == 1:
                            return weights[order[0]], {order[0]}
                        elif fix_start == 0:
                            return 0.0, set()
                        else:
                            return min(0.0, weights[order[0]]), set() # 0.0, set() is better if no edge
                
                    INF = float('inf')
                    # dp0, dp1 lists
                    dp0 = [None] * len(order)
                    dp1 = [None] * len(order)
                    
                    if fix_start == 1:
                        dp0[0] = (INF, set())
                        dp1[0] = (weights[order[0]], {order[0]})
                    elif fix_start == 0:
                        dp0[0] = (0.0, set())
                        dp1[0] = (INF, set())
                    else:
                        dp0[0] = (0.0, set())
                        dp1[0] = (weights[order[0]], {order[0]})
                        
                    for i in range(1, len(order)):
                        w = weights[order[i]]
                        node = order[i]
                        
                        # dp0[i] from dp1[i-1]
                        c, s = dp1[i-1]
                        if c < INF:
                            dp0[i] = (c, set(s))
                        else:
                            dp0[i] = (INF, set())
                            
                        # dp1[i] from min(dp0[i-1], dp1[i-1])
                        c0, s0 = dp0[i-1]
                        c1, s1 = dp1[i-1]
                        
                        if c0 <= c1:
                            if c0 < INF:
                                dp1[i] = (c0 + w, set(s0) | {node})
                            else:
                                dp1[i] = (INF, set())
                        else:
                            if c1 < INF:
                                dp1[i] = (c1 + w, set(s1) | {node})
                            else:
                                dp1[i] = (INF, set())
                                
                    # Apply end constraint
                    if fix_end == 1:
                        return dp1[-1]
                    elif fix_end == 0:
                        return dp0[-1]
                    else:
                        if dp0[-1][0] <= dp1[-1][0]:
                            return dp0[-1]
                        else:
                            return dp1[-1]

                cost_in, nodes_in = dp_path_constrained(order, fix_start=1)
                cost_out, nodes_out = dp_path_constrained(order, fix_start=0, fix_end=1)
                
                if cost_in <= cost_out:
                    total_cost += cost_in
                    total_nodes.update(nodes_in)
                else:
                    total_cost += cost_out
                    total_nodes.update(nodes_out)
            else:
                endpoints = [n for n in component_nodes if deg[n] == 1]
                if endpoints:
                    start_node = endpoints[0]
                    order = [start_node]
                    prev = None
                    curr = start_node
                    while True:
                        nexts = [n for n in adj[curr] if n != prev]
                        if not nexts:
                            break
                        next_node = nexts[0]
                        order.append(next_node)
                        prev = curr
                        curr = next_node
                        
                    cost, nodes = dp_path(order)
                    total_cost += cost
                    total_nodes.update(nodes)
                    
        return total_cost, total_nodes

    for x in range(1, N + 1):
        conflict_pairs = []
        involved_nodes = set()
        
        for b in S:
            a = 2 * b - x
            if a > 0 and a < b and a in S:
                conflict_pairs.append((a, b))
                involved_nodes.add(a)
                involved_nodes.add(b)
                
        if not conflict_pairs:
            S.add(x)
            continue
            
        weights = {y: 1.0 / y for y in involved_nodes}
        min_cover_weight, nodes_to_remove = solve_vertex_cover(involved_nodes, conflict_pairs, weights)
        
        if min_cover_weight < 1.0 / x:
            S.difference_update(nodes_to_remove)
            S.add(x)
            
    return list(S)
