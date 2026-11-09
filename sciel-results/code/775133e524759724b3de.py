import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_longest_cycle(adj, n):
        """Find a long cycle in the graph using DFS with backtracking (limited)."""
        best_cycle = None
        best_len = 0
        
        # Try to find a cycle starting from each vertex
        for start in range(n):
            if not adj[start]:
                continue
            # DFS to find a cycle back to start
            visited = [False] * n
            path = []
            
            def dfs(v, depth):
                nonlocal best_cycle, best_len
                if time.time() > DEADLINE - 10:
                    return
                if depth > 50:  # limit depth to avoid excessive search
                    return
                visited[v] = True
                path.append(v)
                for nb in adj[v]:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        visited[v] = False
                        path.pop()
                        return
                    if not visited[nb]:
                        dfs(nb, depth + 1)
                        if time.time() > DEADLINE - 10:
                            return
                visited[v] = False
                path.pop()
            
            dfs(start, 0)
            if time.time() > DEADLINE - 10:
                break
        
        return best_cycle
    
    def decompose(edges, n):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        # Build adjacency
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        pieces = []
        remaining = set(edge_set)
        
        # Repeatedly find and remove cycles
        while True:
            if time.time() > DEADLINE - 5:
                break
            
            # Find a cycle in the remaining graph
            # Use a simpler approach: find any cycle via DFS
            cycle = None
            visited = [False] * n
            parent = [-1] * n
            
            def find_cycle_dfs(v, start):
                nonlocal cycle
                visited[v] = True
                for nb in adj[v]:
                    if (min(v, nb), max(v, nb)) not in remaining:
                        continue
                    if nb == start and v != start:
                        # Found a cycle
                        cyc = [v]
                        cur = v
                        while cur != start:
                            cur = parent[cur]
                            cyc.append(cur)
                        cycle = cyc
                        return True
                    if not visited[nb]:
                        parent[nb] = v
                        if find_cycle_dfs(nb, start):
                            return True
                return False
            
            # Try to find a cycle starting from each vertex
            found_cycle = False
            for s in range(n):
                if not adj[s]:
                    continue
                visited = [False] * n
                parent = [-1] * n
                if find_cycle_dfs(s, s):
                    found_cycle = True
                    break
                if time.time() > DEADLINE - 5:
                    break
            
            if not found_cycle or cycle is None:
                break
            
            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge in remaining:
                    remaining.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            
            pieces.append(cycle)
        
        # Remaining edges become single-edge pieces
        for u, v in remaining:
            pieces.append([u, v])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > DEADLINE - 5:
            # Fallback: all edges as single pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        pieces = decompose(edges, n)
        results.append(pieces)
    
    return results
