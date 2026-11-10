"""Rewritten program: decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set):
        """Return the set of edges in the 2-core of the graph defined by edge_set."""
        if not edge_set:
            return set()
        
        # Build degree map and adjacency
        deg = defaultdict(int)
        adj = defaultdict(set)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            
        # Queue vertices with degree < 2
        q = deque([v for v in deg if deg[v] < 2])
        in_core = {v: True for v in deg}
        
        while q:
            v = q.popleft()
            if not in_core[v]:
                continue
            in_core[v] = False
            for nb in adj[v]:
                if in_core[nb]:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        q.append(nb)
                        
        # Reconstruct core edges
        core_edges = set()
        for u, v in edge_set:
            if in_core[u] and in_core[v]:
                core_edges.add((u, v))
        return core_edges

    def find_cycle_in_core(core_edges, n):
        """Find a simple cycle in the subgraph defined by core_edges.
        Returns list of vertices or None."""
        if not core_edges:
            return None
            
        adj = defaultdict(set)
        for u, v in core_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Pick a random start vertex
        start = random.choice(list(adj.keys()))
        
        visited = {start}
        parent = {start: None}
        stack = [start]
        
        while stack:
            u = stack[-1]
            nbrs = list(adj[u])
            random.shuffle(nbrs)
            
            advanced = False
            for nb in nbrs:
                if nb == start and len(stack) >= 2:
                    # Cycle found
                    return stack[:]
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = u
                    stack.append(nb)
                    advanced = True
                    break
                    
            if not advanced:
                stack.pop()
                
        return None

    def decompose(edges, n, time_budget):
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        best_pieces = None
        best_count = len(edge_set) + 1
        
        # Number of restarts depends on remaining time and graph size
        # For n=30, this is very fast. We can do many.
        max_restarts = 500 
        
        for r in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            rem = set(edge_set)
            pieces = []
            
            while rem:
                core = get_2_core(rem)
                
                if not core:
                    # Remaining edges are a forest
                    for e in rem:
                        pieces.append(list(e))
                    break
                
                # Try to find a cycle in the core
                # We might try a few times if the first one is short? 
                # Or just take the first one found by randomized DFS.
                cycle = find_cycle_in_core(core, n)
                
                if cycle:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        rem.discard(e)
                    pieces.append(cycle[:])
                else:
                    # Should not happen if core is not empty, but fallback
                    e = tuple(next(iter(rem)))
                    rem.discard(e)
                    pieces.append(list(e))
                    
            cnt = len(pieces)
            if cnt < best_count:
                best_count = cnt
                best_pieces = [p[:] for p in pieces]
                
        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]
            
        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 5:
            # Fallback: all edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
