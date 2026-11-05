import random
from collections import defaultdict, deque

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_path(remaining_edges):
        if not remaining_edges:
            return []
        
        # Build adjacency for remaining edges
        adj = defaultdict(set)
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_path = []
        
        # Try starting from each vertex
        # Optimization: Only try vertices with degree > 0
        candidates = [v for v in adj if adj[v]]
        
        for start in candidates:
            # DFS to find a long simple path
            # Heuristic: Visit neighbors in order of decreasing degree in the remaining graph
            # We use a stack for iterative DFS to avoid recursion limits, though N=30 is safe for recursion
            # Let's use a recursive DFS with memoization or just simple backtracking?
            # Simple backtracking for longest path is too slow.
            # Greedy heuristic:
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Get neighbors of current in remaining graph
                neighbors = [n for n in adj[current] if n not in visited]
                if not neighbors:
                    break
                
                # Sort neighbors by degree in remaining graph (descending)
                # This is Warnsdorff's rule for Hamiltonian path, adapted for longest path
                neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                
                next_node = neighbors[0]
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if len(path) > len(best_path):
                best_path = path
                
        return best_path

    def decompose(edges):
        remaining = set(edges)
        paths = []
        
        while remaining:
            path = find_longest_path(remaining)
            if not path:
                # Fallback: should not happen if remaining is not empty
                # Pick any edge
                u, v = next(iter(remaining))
                path = [u, v]
            
            # Remove edges from remaining
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
            
            paths.append(path)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
