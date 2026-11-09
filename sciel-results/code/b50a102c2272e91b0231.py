import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_longest_path(adj, available_edges, rng):
        """
        Find a long simple path in the graph defined by adj and available_edges.
        Uses randomized DFS to explore different paths.
        """
        if not available_edges:
            return []
        
        # Pick a random start vertex
        # Prefer vertices with high degree in available edges
        degrees = defaultdict(int)
        for u, v in available_edges:
            degrees[u] += 1
            degrees[v] += 1
        
        if not degrees:
            return []
            
        # Sort vertices by degree descending, add some randomness
        verts = list(degrees.keys())
        # Shuffle to break ties randomly
        rng.shuffle(verts)
        verts.sort(key=lambda x: -degrees[x])
        
        best_path = []
        
        # Try starting from a few high-degree vertices
        candidates = verts[:min(5, len(verts))]
        
        for start in candidates:
            # DFS
            path = [start]
            visited = {start}
            
            def dfs(u):
                nonlocal path
                # Get neighbors in available edges
                neighbors = []
                for v in adj[u]:
                    e = tuple(sorted((u, v)))
                    if e in available_edges and v not in visited:
                        neighbors.append(v)
                
                if not neighbors:
                    return
                
                # Shuffle neighbors
                rng.shuffle(neighbors)
                
                for v in neighbors:
                    visited.add(v)
                    path.append(v)
                    dfs(v)
                    # Backtrack
                    path.pop()
                    visited.remove(v)
                    # We don't prune aggressively here, we just want one long path.
                    # To make it faster, we could stop if we reach a certain length,
                    # but for N=30, full DFS is okay if we limit the depth or branches.
                    # Actually, full DFS on 30 nodes is fine.
            
            dfs(start)
            
            if len(path) > len(best_path):
                best_path = path
        
        return best_path

    def merge_paths(paths, edge_set):
        """
        Try to merge paths to reduce the count.
        """
        if len(paths) <= 1:
            return paths
        
        improved = True
        while improved:
            improved = False
            new_paths = []
            merged_indices = set()
            
            for i in range(len(paths)):
                if i in merged_indices:
                    continue
                path_i = paths[i]
                merged = False
                
                for j in range(i + 1, len(paths)):
                    if j in merged_indices:
                        continue
                    path_j = paths[j]
                    
                    # Try merging: path_i + path_j
                    if path_i[-1] == path_j[0]:
                        candidate = path_i + path_j[1:]
                        if len(set(candidate)) == len(candidate):
                            # Check edges
                            valid = True
                            for k in range(len(candidate) - 1):
                                e = tuple(sorted((candidate[k], candidate[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(j)
                                improved = True
                                merged = True
                                break
                    
                    # Try merging: path_j + path_i
                    elif path_j[-1] == path_i[0]:
                        candidate = path_j + path_i[1:]
                        if len(set(candidate)) == len(candidate):
                            valid = True
                            for k in range(len(candidate) - 1):
                                e = tuple(sorted((candidate[k], candidate[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(i)
                                improved = True
                                merged = True
                                break
                    
                    # Try reversing path_j
                    path_j_rev = path_j[::-1]
                    if path_i[-1] == path_j_rev[0]:
                        candidate = path_i + path_j_rev[1:]
                        if len(set(candidate)) == len(candidate):
                            valid = True
                            for k in range(len(candidate) - 1):
                                e = tuple(sorted((candidate[k], candidate[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(j)
                                improved = True
                                merged = True
                                break
                                
                    if path_j_rev[-1] == path_i[0]:
                        candidate = path_j_rev + path_i[1:]
                        if len(set(candidate)) == len(candidate):
                            valid = True
                            for k in range(len(candidate) - 1):
                                e = tuple(sorted((candidate[k], candidate[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                new_paths.append(candidate)
                                merged_indices.add(i)
                                improved = True
                                merged = True
                                break
                            
                if not merged:
                    new_paths.append(path_i)
            
            paths = new_paths
        
        return paths

    def solve_graph(edges, seed):
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        rng = random.Random(seed + 100)
        
        paths = []
        available = set(edge_set)
        
        # Greedy longest path decomposition
        while available:
            if time.time() > deadline:
                break
            
            # Find a long path
            path = find_longest_path(adj, available, rng)
            
            if not path:
                # Fallback: pick any edge
                u, v = next(iter(available))
                path = [u, v]
            
            # Remove edges from available
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e in available:
                    available.remove(e)
            
            paths.append(path)
        
        # If we ran out of time and still have edges, add them as single-edge paths
        if available:
            for u, v in available:
                paths.append([u, v])
        
        # Try to merge
        if time.time() < deadline - 5:
            paths = merge_paths(paths, edge_set)
            
        return paths

    results = []
    for i, spec in enumerate([(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]):
        if time.time() > deadline:
            # Fallback
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = solve_graph(edges, spec[2])
        results.append(paths)
        
    return results
