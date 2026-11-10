"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def bfs_path(adj, src, dst, remaining):
        """Find a path from src to dst using only edges in remaining."""
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adj.get(node, [])):
                edge = tuple(sorted((node, neighbor)))
                if edge not in remaining or neighbor in parent:
                    continue
                parent[neighbor] = node
                if neighbor == dst:
                    path = [dst]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                queue.append(neighbor)
        return None

    def extend_path(adj, remaining, path):
        """Extend a path at both ends using available edges."""
        # Extend from start
        while True:
            current = path[0]
            extended = False
            for neighbor in sorted(adj.get(current, [])):
                e = tuple(sorted((current, neighbor)))
                if e in remaining:
                    path.insert(0, neighbor)
                    extended = True
                    break
            if not extended:
                break
        # Extend from end
        while True:
            current = path[-1]
            extended = False
            for neighbor in sorted(adj.get(current, [])):
                e = tuple(sorted((current, neighbor)))
                if e in remaining:
                    path.append(neighbor)
                    extended = True
                    break
            if not extended:
                break
        return path

    def decompose_paths(edges):
        if not edges:
            return []
        
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        
        while remaining:
            if time.time() > DEADLINE:
                for e in sorted(remaining):
                    paths.append(list(e))
                return paths
            
            # Compute degrees in remaining graph
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            odd_verts = sorted([v for v in deg if deg[v] % 2 == 1])
            
            if not odd_verts:
                # All degrees even: find an Eulerian circuit
                start = None
                for v in deg:
                    if deg[v] > 0:
                        start = v
                        break
                if start is None:
                    break
                
                # Hierholzer's algorithm
                stack = [start]
                circuit = []
                while stack:
                    v = stack[-1]
                    if adj.get(v):
                        w = next(iter(adj[v]))
                        edge = tuple(sorted((v, w)))
                        remaining.discard(edge)
                        adj[v].discard(w)
                        adj[w].discard(v)
                        stack.append(w)
                    else:
                        circuit.append(stack.pop())
                
                circuit.reverse()
                if circuit:
                    paths.append(circuit)
            else:
                # Pair up odd vertices and find paths between each pair
                # Greedy: pair consecutive odd vertices
                pairing = [(odd_verts[i], odd_verts[i + 1]) 
                          for i in range(0, len(odd_verts), 2)]
                
                for a, b in pairing:
                    if time.time() > DEADLINE:
                        break
                    path = bfs_path(adj, a, b, remaining)
                    if path is None:
                        continue
                    path = extend_path(adj, remaining, path)
                    
                    # Remove edges from remaining
                    for i in range(len(path) - 1):
                        e = tuple(sorted((path[i], path[i + 1])))
                        remaining.discard(e)
                        if path[i + 1] in adj[path[i]]:
                            adj[path[i]].discard(path[i + 1])
                        if path[i] in adj[path[i + 1]]:
                            adj[path[i + 1]].discard(path[i])
                    
                    paths.append(path)
        
        # Handle any remaining edges
        if remaining:
            for e in sorted(remaining):
                paths.append(list(e))
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        paths = decompose_paths(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
