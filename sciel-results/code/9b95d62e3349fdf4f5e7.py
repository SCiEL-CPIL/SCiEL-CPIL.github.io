"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict, deque

def construct():
    # Replicate the verifier's instance generator exactly.
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_bridges(edges):
        """
        Find all bridges in the graph defined by the set of edges.
        Returns a set of bridge edges.
        """
        if not edges:
            return set()
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        visited = set()
        bridges = set()
        
        # Tarjan's bridge-finding algorithm
        time_counter = [0]
        disc = {}
        low = {}
        
        def dfs(u, parent):
            disc[u] = time_counter[0]
            low[u] = time_counter[0]
            time_counter[0] += 1
            visited.add(u)
            
            for v in adj[u]:
                if v == parent:
                    continue
                if v not in disc:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.add(tuple(sorted((u, v))))
                else:
                    low[u] = min(low[u], disc[v])
        
        for node in adj:
            if node not in visited:
                dfs(node, -1)
                
        return bridges

    def decompose(edges):
        if not edges:
            return []
        
        remaining = set(edges)
        pieces = []
        
        def find_cycle(rem_edges):
            if not rem_edges:
                return None
            adj = defaultdict(set)
            for u, v in rem_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            active_nodes = [n for n in adj if adj[n]]
            if not active_nodes:
                return None
                
            visited = set()
            parent = {}
            
            def dfs_cycle(node, start_node):
                visited.add(node)
                for neighbor in sorted(adj[node]):
                    if neighbor == start_node and len(visited) >= 3:
                        path = [node]
                        cur = node
                        while cur != start_node:
                            cur = parent[cur]
                            path.append(cur)
                        path.reverse()
                        return path
                    if neighbor not in visited:
                        parent[neighbor] = node
                        res = dfs_cycle(neighbor, start_node)
                        if res:
                            return res
                return None

            for start_node in active_nodes:
                if start_node in visited:
                    continue
                # Since we only call this when no bridges exist, 
                # any component with edges must contain a cycle.
                res = dfs_cycle(start_node, start_node)
                if res:
                    return res
            return None

        while remaining:
            # 1. Identify and remove bridges (they cannot be in any cycle)
            bridges = find_bridges(remaining)
            if bridges:
                for b in bridges:
                    pieces.append(list(b))
                    remaining.remove(b)
                # Removing bridges may create new bridges, so loop continues
                continue
            
            # 2. No bridges, so find a cycle
            cycle = find_cycle(remaining)
            if cycle:
                pieces.append(cycle)
                # Remove cycle edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge in remaining:
                        remaining.remove(edge)
            else:
                # Safety fallback: should not happen if logic is correct
                for e in remaining:
                    pieces.append(list(e))
                break
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
