import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def greedy_path_cover_smart(edges, rng=None):
        """
        Smart greedy path cover:
        1. Start from degree-1 vertices (forced paths)
        2. When extending, prefer neighbors that lead to longer paths
        3. Use lookahead to avoid dead ends
        """
        if not edges:
            return []
        
        if rng is None:
            rng = random.Random(42)
            
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        while remaining_edges:
            # Find a start vertex: prefer degree 1, then degree 2, then random
            degree1 = [u for u in adj if len(adj[u]) == 1]
            degree2 = [u for u in adj if len(adj[u]) == 2]
            higher = [u for u in adj if len(adj[u]) > 2]
            
            if degree1:
                start = rng.choice(degree1)
            elif degree2:
                start = rng.choice(degree2)
            elif higher:
                start = rng.choice(higher)
            else:
                break
                
            path = [start]
            visited_vertices = {start}
            current = start
            
            while True:
                # Get valid neighbors
                valid_neighbors = []
                for neighbor in adj[current]:
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in remaining_edges and neighbor not in visited_vertices:
                        valid_neighbors.append(neighbor)
                        
                if not valid_neighbors:
                    break
                
                # Choose next vertex with lookahead: prefer neighbors with more unvisited neighbors
                if len(valid_neighbors) == 1:
                    next_v = valid_neighbors[0]
                else:
                    # Score each neighbor by how many unvisited neighbors they have
                    def score(neighbor):
                        unvisited_count = 0
                        for nb in adj[neighbor]:
                            if nb not in visited_vertices and nb != current:
                                edge = (min(neighbor, nb), max(neighbor, nb))
                                if edge in remaining_edges:
                                    unvisited_count += 1
                        return unvisited_count
                    
                    # Prefer higher score (more options ahead)
                    best_score = -1
                    best_candidates = []
                    for v in valid_neighbors:
                        s = score(v)
                        if s > best_score:
                            best_score = s
                            best_candidates = [v]
                        elif s == best_score:
                            best_candidates.append(v)
                    next_v = rng.choice(best_candidates)
                    
                path.append(next_v)
                visited_vertices.add(next_v)
                current = next_v
                
            if len(path) > 1:
                paths.append(path)
                
            # Remove edges used in this path
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    adj[u].remove(v)
                    adj[v].remove(u)
                
            if time.time() > deadline:
                break
                
        return paths

    def try_merge_paths(paths, adj, remaining_edges):
        """
        Try to merge two paths by finding a connecting edge.
        Returns True if merge was successful.
        """
        if len(paths) < 2:
            return False
        
        # For each pair of paths, check if we can connect end of one to start of another
        for i in range(len(paths)):
            for j in range(len(paths)):
                if i == j:
                    continue
                
                # Try connecting end of path i to start of path j
                end_i = paths[i][-1]
                start_j = paths[j][0]
                
                edge = (min(end_i, start_j), max(end_i, start_j))
                if edge in remaining_edges:
                    # Check if merging would create a cycle (vertex already in path j)
                    if end_i not in set(paths[j]):
                        # Merge: extend path i with path j
                        paths[i] = paths[i] + paths[j]
                        # Remove the connecting edge
                        remaining_edges.remove(edge)
                        adj[end_i].remove(start_j)
                        adj[start_j].remove(end_i)
                        # Remove path j
                        paths.pop(j)
                        return True
                        
                # Try connecting start of path i to end of path j
                start_i = paths[i][0]
                end_j = paths[j][-1]
                
                edge = (min(start_i, end_j), max(start_i, end_j))
                if edge in remaining_edges:
                    # Check if merging would create a cycle
                    if end_j not in set(paths[i]):
                        # Merge: prepend path j to path i (reversed)
                        paths[i] = list(reversed(paths[j])) + paths[i]
                        # Remove the connecting edge
                        remaining_edges.remove(edge)
                        adj[start_i].remove(end_j)
                        adj[end_j].remove(start_i)
                        # Remove path j
                        paths.pop(j)
                        return True
        
        return False

    def local_search_improve(paths, edges):
        """
        Local search: try to merge paths iteratively.
        """
        if not paths:
            return paths
        
        # Build adjacency and remaining edges
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find which edges are used in paths
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                used_edges.add(edge)
        
        # Remaining edges are those not in any path
        all_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        remaining_edges = all_edges - used_edges
        
        # Try to merge paths
        improved = True
        while improved:
            improved = False
            for _ in range(100):  # Try up to 100 merges
                if try_merge_paths(paths, adj, remaining_edges):
                    improved = True
                    if time.time() > deadline:
                        return paths
        
        return paths

    def solve_graph(edges):
        best_cover = None
        best_count = float('inf')
        
        # Run smart greedy with different random seeds
        for seed in range(20):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            cover = greedy_path_cover_smart(edges, rng)
            
            # Apply local search improvement
            cover = local_search_improve(cover, edges)
            
            count = len(cover)
            if count < best_count:
                best_count = count
                best_cover = cover
        
        if best_cover is None:
            # Fallback: trivial cover
            best_cover = [[list(e)] for e in edges]
            
        return best_cover

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        edges = instance(*spec)
        cover = solve_graph(edges)
        results.append(cover)
        
    return results
