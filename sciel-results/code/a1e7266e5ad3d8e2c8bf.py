import time
import random
from collections import defaultdict, deque
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        # The seed program creates a path on the shuffled order
        # zip(order, order[1:]) gives (order[0], order[1]), (order[1], order[2]), ...
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        # Then adds random edges
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_2_factor_cycles(adj, n):
        """
        Finds a 2-factor (set of vertex-disjoint cycles covering all vertices) 
        using max bipartite matching (Hopcroft-Karp). 
        Returns list of cycles (list of vertices) and the set of edges used.
        """
        pair_u = [-1] * n
        pair_v = [-1] * n
        dist = [0] * n
        
        def bfs():
            q = deque()
            for u in range(n):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        if matching < n:
            return [], set()
            
        # Extract cycles from the perfect matching
        visited = [False] * n
        cycles = []
        edges_used = set()
        
        for i in range(n):
            if not visited[i]:
                cycle = []
                u = i
                while not visited[u]:
                    visited[u] = True
                    cycle.append(u)
                    v = pair_u[u]
                    edge = tuple(sorted((u, v)))
                    edges_used.add(edge)
                    u = v
                cycles.append(cycle)
        
        return cycles, edges_used

    def find_cycle_heuristic(adj, n, rng):
        """
        Finds a cycle in the graph using randomized DFS.
        Returns a list of vertices forming the cycle, or None.
        """
        # Try a few random starts to find a reasonably long cycle
        # We limit the search depth implicitly by the graph size
        
        # To improve speed, we can just do one DFS from a random node
        # If the graph is dense, a cycle is found quickly.
        
        start = rng.randint(0, n - 1)
        if not adj[start]:
            return None
            
        stack = [(start, [start])]
        visited = {start}
        
        while stack:
            node, path = stack.pop()
            neighbors = list(adj[node])
            rng.shuffle(neighbors)
            
            for nb in neighbors:
                if nb == path[0] and len(path) >= 3:
                    return path
                elif nb not in visited:
                    visited.add(nb)
                    stack.append((nb, path + [nb]))
                    
        return None

    def decompose(edges, n):
        pieces = []
        remaining = set(edges)
        
        # Build adjacency for initial graph
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Phase 1: 2-Factor
        if time.time() < deadline:
            cycles, used_edges = find_2_factor_cycles(adj, n)
            for cycle in cycles:
                pieces.append(cycle)
            remaining -= used_edges
            
            # Update adj for remaining
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)

        # Phase 2: Greedy Cycle Extraction
        rng = Random(42)
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Find a cycle in remaining
            cycle = find_cycle_heuristic(adj, n, rng)
            
            if cycle:
                pieces.append(cycle)
                # Remove edges from remaining
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    remaining.discard(edge)
                    # Update adj
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                # No cycle found, take one edge
                edge = next(iter(remaining))
                pieces.append(list(edge))
                remaining.discard(edge)
                u, v = edge
                adj[u].discard(v)
                adj[v].discard(u)
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)
        
    return results
