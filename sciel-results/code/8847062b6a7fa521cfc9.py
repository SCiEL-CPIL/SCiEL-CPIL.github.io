"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges, n, deadline):
    """Decompose graph edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deg = {u: len(adj[u]) for u in range(n)}
    pieces = []
    
    def find_cycle():
        """Find a cycle in the remaining graph using DFS from high-degree start."""
        # Try multiple random starting points, prefer high-degree
        candidates = [u for u in range(n) if deg[u] >= 2]
        if not candidates:
            return None
        
        # Sort by degree descending, shuffle ties
        import random
        rng = random.Random()
        candidates.sort(key=lambda u: -deg[u])
        
        for start in candidates:
            if deg[start] < 2:
                continue
            # DFS to find a cycle back to start
            stack = [(start, start, [start])]
            visited = {start}
            parent = {start: None}
            
            while stack:
                node, prev, path = stack.pop()
                for nb in adj[node]:
                    if nb not in remaining.get((min(node, nb), max(node, nb)), False) and \
                       (min(node, nb), max(node, nb)) not in remaining:
                        continue
                    # Check edge exists in remaining
                    e = tuple(sorted((node, nb)))
                    if e not in remaining:
                        continue
                    if nb == start and len(path) >= 3:
                        return path
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = node
                        stack.append((nb, node, path + [nb]))
            # Reset visited for next start
            visited.clear()
        
        return None
    
    # Better: use a proper DFS that tracks the actual path
    def find_cycle_v2():
        """Find a cycle using iterative DFS with explicit path tracking."""
        import random
        rng = random.Random()
        
        candidates = [u for u in range(n) if deg[u] >= 2]
        if not candidates:
            return None
        
        # Try from highest degree first
        candidates.sort(key=lambda u: -deg[u])
        
        for start in candidates:
            if deg[start] < 2:
                continue
            
            # DFS: track path
            visited = set()
            stack = [(start, [start])]
            visited.add(start)
            
            while stack:
                node, path = stack.pop()
                for nb in sorted(adj[node], key=lambda x: -deg[x]):
                    e = tuple(sorted((node, nb)))
                    if e not in remaining:
                        continue
                    if nb == start and len(path) >= 3:
                        return path
                    if nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
            
            visited.clear()
        
        return None
    
    # Even better: proper cycle finding with backtracking
    def find_cycle_v3():
        """Find any cycle in remaining graph."""
        for start in range(n):
            if deg[start] < 2:
                continue
            # BFS/DFS from start looking for any cycle
            # Use color: 0=white, 1=gray, 2=black
            color = {u: 0 for u in range(n)}
            parent = {}
            
            def dfs(u):
                color[u] = 1
                for v in adj[u]:
                    e = tuple(sorted((u, v)))
                    if e not in remaining:
                        continue
                    if color[v] == 0:
                        parent[v] = u
                        result = dfs(v)
                        if result:
                            return result
                    elif color[v] == 1:
                        # Found a back edge, extract cycle
                        cycle = [v, u]
                        curr = u
                        while curr != v:
                            curr = parent[curr]
                            cycle.append(curr)
                        cycle.reverse()
                        if len(set(cycle)) >= 3:
                            return cycle
                color[u] = 2
                return None
            
            result = dfs(start)
            if result:
                return result
        
        return None
    
    # Use a simpler and more reliable approach:
    # Repeatedly find a cycle by doing DFS from each unvisited vertex
    def find_one_cycle():
        """Find one cycle in the remaining edge set."""
        # Build adjacency from remaining
        rem_adj = defaultdict(set)
        for e in remaining:
            u, v = e
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Find a vertex with degree >= 2
        for u in range(n):
            if len(rem_adj[u]) < 2:
                continue
            # DFS from u
            visited = set()
            stack = [(u, [u])]
            visited.add(u)
            
            while stack:
                node, path = stack.pop()
                for nb in rem_adj[node]:
                    if nb == u and len(path) >= 3:
                        return path
                    if nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
            # Reset
            visited = set()
        
        return None
    
    # Main loop: extract cycles greedily
    while remaining:
        if time.time() > deadline:
            break
        
        cycle = find_one_cycle()
        if cycle:
            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in remaining:
                    remaining.remove(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    deg[u] = len(adj[u])
                    deg[v] = len(adj[v])
            pieces.append(cycle)
        else:
            # No cycle left, take one edge
            e = next(iter(remaining))
            remaining.remove(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] = len(adj[u])
            deg[v] = len(adj[v])
            pieces.append(list(e))
    
    # Add any remaining edges as single-edge pieces
    for e in remaining:
        pieces.append(list(e))
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = _instance(n, p, seed)
        
        if not edges:
            results.append([])
            continue
        
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        deg = {u: len(adj[u]) for u in range(n)}
        pieces = []
        
        def find_one_cycle():
            rem_adj = defaultdict(set)
            for e in remaining:
                u, v = e
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            for u in range(n):
                if len(rem_adj[u]) < 2:
                    continue
                visited = set()
                stack = [(u, [u])]
                visited.add(u)
                while stack:
                    node, path = stack.pop()
                    for nb in rem_adj[node]:
                        if nb == u and len(path) >= 3:
                            return path
                        if nb not in visited:
                            visited.add(nb)
                            stack.append((nb, path + [nb]))
                visited = set()
            return None
        
        while remaining:
            if time.time() > deadline:
                break
            cycle = find_one_cycle()
            if cycle:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e in remaining:
                        remaining.remove(e)
                pieces.append(cycle)
            else:
                e = next(iter(remaining))
                remaining.remove(e)
                pieces.append(list(e))
        
        for e in remaining:
            pieces.append(list(e))
        
        results.append(pieces)
    
    return results
