import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, time_budget):
        """Decompose graph into cycles and edges, minimizing number of pieces."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining = set(edges)
        
        # Repeatedly find a long cycle in the remaining graph
        while remaining and time.time() < time_budget:
            # Find a cycle using DFS
            best_cycle = None
            best_len = 0
            
            # Try starting from each vertex with remaining edges
            for start in range(n):
                if not adj[start]:
                    continue
                
                # DFS to find a cycle
                visited = {start: 0}
                parent = {start: None}
                in_stack = {start}
                
                def dfs(u):
                    nonlocal best_cycle, best_len
                    for v in sorted(adj[u]):
                        e = (min(u, v), max(u, v))
                        if e not in remaining:
                            continue
                        if v not in visited:
                            visited[v] = visited[u] + 1
                            parent[v] = u
                            in_stack.add(v)
                            dfs(v)
                            in_stack.remove(v)
                        elif v in in_stack:
                            length = visited[u] - visited[v] + 1
                            if length >= 3 and length > best_len:
                                # Reconstruct cycle
                                cycle = [u]
                                cur = u
                                while cur != v:
                                    cur = parent[cur]
                                    cycle.append(cur)
                                best_cycle = cycle
                                best_len = length
                
                dfs(start)
                
                if best_cycle and len(best_cycle) > best_len:
                    best_len = len(best_cycle)
                
                if time.time() > time_budget:
                    break
            
            if best_cycle:
                pieces.append(best_cycle)
                # Remove edges of this cycle
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        remaining.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
            else:
                # No more cycles, add remaining edges as singles
                for e in remaining:
                    pieces.append(list(e))
                break
        
        # Add any remaining edges as singles
        for e in remaining:
            pieces.append(list(e))
        
        return pieces

    def decompose_graph_v2(edges, n, time_budget):
        """Improved decomposition using multiple strategies."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining = set(edges)
        
        # Strategy: repeatedly find longest cycle
        while remaining and time.time() < time_budget:
            best_cycle = None
            best_len = 0
            
            # Try to find a long cycle by exploring from high-degree vertices first
            vertices_by_degree = sorted(range(n), key=lambda x: -len(adj[x]))
            
            for start in vertices_by_degree:
                if not adj[start]:
                    continue
                
                # BFS/DFS to find a cycle back to start
                # Use iterative DFS with path tracking
                stack = [(start, [start])]
                visited = {start}
                
                while stack and time.time() < time_budget:
                    node, path = stack.pop()
                    for v in sorted(adj[node], key=lambda x: -len(adj[x])):
                        e = (min(node, v), max(node, v))
                        if e not in remaining:
                            continue
                        if v == start and len(path) >= 3:
                            # Found a cycle
                            cycle = path
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle
                        elif v not in visited:
                            visited.add(v)
                            stack.append((v, path + [v]))
                
                if best_cycle and len(best_cycle) > best_len:
                    best_len = len(best_cycle)
                
                if time.time() > time_budget:
                    break
            
            if best_cycle:
                pieces.append(best_cycle)
                # Remove edges of this cycle
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        remaining.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
            else:
                # No more cycles, add remaining edges as singles
                for e in remaining:
                    pieces.append(list(e))
                break
        
        # Add any remaining edges as singles
        for e in remaining:
            pieces.append(list(e))
        
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    # Allocate time budget for each graph
    total_time = deadline - start_time
    time_per_graph = total_time / 3
    
    for i, spec in enumerate(specs):
        edges = instance(*spec)
        n = spec[0]
        
        # Give more time to later graphs if we have time to spare
        current_time = time.time()
        remaining_time = deadline - current_time
        graphs_remaining = len(specs) - i
        time_budget = min(remaining_time / graphs_remaining, time_per_graph * 1.5)
        
        # Use the better decomposition strategy
        pieces = decompose_graph_v2(edges, n, current_time + time_budget)
        results.append(pieces)
    
    return results
