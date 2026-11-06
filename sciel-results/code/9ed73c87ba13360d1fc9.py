# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    deadline = t0 + 130  # safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles_in_graph(n, edge_set):
        """Find edge-disjoint cycles greedily, preferring longer cycles."""
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = []
        used_edges = set()
        
        # Precompute all simple cycles up to a reasonable length using DFS
        # For efficiency, we'll do iterative DFS looking for cycles
        
        def find_longest_cycle(start, max_depth=30):
            """Find a cycle starting and ending at start vertex using DFS."""
            best = None
            # Try to find cycles through start
            visited = {start}
            path = [start]
            
            def dfs(v, depth):
                nonlocal best
                if time.time() > deadline:
                    return
                if depth >= max_depth:
                    return
                for nb in sorted(adj[v]):
                    if nb == start and len(path) >= 3:
                        cycle = list(path)
                        if best is None or len(cycle) > len(best):
                            best = cycle
                    elif nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, depth + 1)
                        path.pop()
                        visited.discard(nb)
            
            dfs(start, 1)
            return best
        
        # Iteratively find cycles
        remaining = set(edge_set)
        
        while True:
            if time.time() > deadline - 5:
                break
            
            # Find the longest cycle in remaining edges
            best_cycle = None
            # Try from each vertex
            for v in sorted(adj.keys()):
                if not adj[v]:
                    continue
                # Check if v still has edges in remaining
                has_edge = False
                for nb in adj[v]:
                    e = tuple(sorted((v, nb)))
                    if e in remaining:
                        has_edge = True
                        break
                if not has_edge:
                    continue
                
                # Build temporary adjacency for remaining edges
                temp_adj = defaultdict(set)
                for u, v2 in remaining:
                    temp_adj[u].add(v2)
                    temp_adj[v2].add(u)
                
                # DFS to find cycle through v
                visited = {v}
                path = [v]
                
                def dfs2(cur, depth):
                    nonlocal best_cycle
                    if time.time() > deadline - 5:
                        return
                    if depth >= 25:
                        return
                    for nb in sorted(temp_adj[cur]):
                        if nb == v and len(path) >= 3:
                            cycle = list(path)
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle
                        elif nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            dfs2(nb, depth + 1)
                            path.pop()
                            visited.discard(nb)
                
                dfs2(v, 1)
                
                if best_cycle and len(best_cycle) >= 5:
                    break  # Good enough, move on
            
            if not best_cycle:
                break
            
            # Extract edges from cycle
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
            
            cycles.append(best_cycle)
        
        # Remaining edges
        remaining_edges = [list(e) for e in sorted(remaining)]
        
        return cycles, remaining_edges

    def decompose(n, p, seed):
        edges = instance(n, p, seed)
        edge_set = set(tuple(e) for e in edges)
        
        cycles, remaining_edges = find_cycles_in_graph(n, edge_set)
        
        pieces = []
        for c in cycles:
            pieces.append(list(c))
        for e in remaining_edges:
            pieces.append(e)
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 10:
            # Fallback: all edges as pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        pieces = decompose(*spec)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
