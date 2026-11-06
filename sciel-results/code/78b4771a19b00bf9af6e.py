"""Path decomposition of graphs using greedy + randomized restarts with improved heuristics."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    best_results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = _instance(*spec)
            best_results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        result = _decompose(edges, deadline)
        if result is not None:
            best_results.append(result)
        else:
            # Fallback: each edge as its own path
            best_results.append([list(e) for e in edges])
    
    return best_results


def _instance(n, p, seed):
    """Generate the instance graph exactly as in the verifier."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline):
    """Decompose edges into minimum number of edge-disjoint simple paths."""
    if not edges:
        return []
    
    best_paths = None
    best_count = len(edges)  # worst case: each edge is its own path
    
    # Number of restarts based on available time
    num_restarts = max(50, int((deadline - time.time()) * 15))
    num_restarts = min(num_restarts, 300)
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        paths = _greedy_decompose(edges, rng, deadline)
        
        if paths is not None:
            # Post-process: merge consecutive paths that share an endpoint
            paths = _merge_paths(paths, edges)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                # Early stop if we hit theoretical lower bound
                lower_bound = _lower_bound(edges)
                if count <= lower_bound:
                    break
    
    if best_paths is None:
        return [list(e) for e in edges]
    
    return best_paths


def _lower_bound(edges):
    """Compute theoretical lower bound on number of paths."""
    if not edges:
        return 0
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    odd_count = sum(1 for d in degree.values() if d % 2 == 1)
    return max(1, (odd_count + 1) // 2)


def _greedy_decompose(edges, rng, deadline):
    """
    Greedy path decomposition:
    1. Build adjacency from remaining edges
    2. Repeatedly find a long simple path using remaining edges
    3. Remove those edges and repeat
    Prioritize starting from odd-degree vertices.
    """
    remaining = set(edges)
    if not remaining:
        return []
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    degree = defaultdict(int)
    for u, v in remaining:
        degree[u] += 1
        degree[v] += 1
    
    paths = []
    
    while remaining:
        if time.time() > deadline:
            return None
        
        # Find a good starting vertex
        start = _pick_start(adj, degree, rng)
        if start is None:
            # Fallback: pick any vertex with edges
            candidates = [v for v in adj if adj[v]]
            if not candidates:
                break
            start = rng.choice(candidates)
        
        # Find a long simple path starting from start
        path = _find_long_path(start, adj, degree, rng, remaining, deadline)
        
        if path is None or len(path) < 2:
            # Can't find a path; take a single edge
            # Find any edge incident to start or any edge
            if adj[start]:
                next_v = rng.choice(list(adj[start]))
                path = [start, next_v]
            else:
                # Pick any remaining edge
                e = next(iter(remaining))
                path = [e[0], e[1]]
        
        # Extract edges from path and remove them
        path_edges = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = tuple(sorted((u, v)))
            if edge in remaining:
                path_edges.add(edge)
        
        if not path_edges:
            # Shouldn't happen, but safety
            break
        
        for e in path_edges:
            remaining.discard(e)
        
        # Update adjacency and degrees
        for u, v in path_edges:
            adj[u].discard(v)
            adj[v].discard(u)
            degree[u] -= 1
            degree[v] -= 1
        
        # Clean up empty adj entries
        for v in list(adj.keys()):
            if not adj[v]:
                del adj[v]
        
        paths.append(path)
    
    return paths


def _pick_start(adj, degree, rng):
    """Pick a starting vertex, preferring odd-degree vertices."""
    odd_vertices = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
    if odd_vertices:
        return rng.choice(odd_vertices)
    
    # Prefer high-degree vertices
    candidates = [v for v in adj if adj[v]]
    if not candidates:
        return None
    
    # Weight by degree
    weights = [max(1, degree.get(v, 0)) for v in candidates]
    total = sum(weights)
    r = rng.random() * total
    cum = 0
    for v, w in zip(candidates, weights):
        cum += w
        if r <= cum:
            return v
    return candidates[-1]


def _find_long_path(start, adj, degree, rng, remaining, deadline):
    """
    Find a long simple path starting from start using DFS with backtracking.
    Uses randomized exploration to find long paths.
    """
    best_path = [start]
    
    # Use a greedy approach with limited backtracking
    _greedy_extend(start, adj, best_path, rng, deadline)
    
    return best_path if len(best_path) >= 2 else None


def _greedy_extend(start, adj, best_path, rng, deadline):
    """
    Greedy path extension with occasional backtracking.
    """
    path = [start]
    visited = {start}
    
    max_depth = 80  # Allow longer paths
    for step in range(max_depth):
        if time.time() > deadline:
            break
        current = path[-1]
        neighbors = [nb for nb in adj[current] if nb not in visited]
        
        if not neighbors:
            break
        
        # Choose next vertex: combine Warnsdorff's rule with degree consideration
        def score(nb):
            unvisited_count = sum(1 for x in adj[nb] if x not in visited)
            # Prefer vertices with fewer unvisited neighbors (Warnsdorff)
            # but also consider overall degree to avoid isolating high-degree vertices
            deg_bonus = 1.0 / (1.0 + adj[nb].__len__() * 0.1)
            return unvisited_count * 1.0 - deg_bonus * 0.5
        
        # Sometimes pick greedily, sometimes randomly
        if rng.random() < 0.80:
            # Pick the neighbor with best score (Warnsdorff-like)
            neighbors.sort(key=score)
            # Pick from top 2-3 with some randomness
            top_k = min(3, len(neighbors))
            choice = rng.choice(neighbors[:top_k])
        else:
            choice = rng.choice(neighbors)
        
        path.append(choice)
        visited.add(choice)
        
        if len(path) > len(best_path):
            best_path[:] = path[:]
    
    return path


def _merge_paths(paths, edges):
    """
    Post-processing: merge consecutive paths that share an endpoint.
    If path A ends at vertex v and path B starts at vertex v (or vice versa),
    we can merge them into a single path, reducing the count by 1.
    """
    if len(paths) <= 1:
        return paths
    
    # Build a map from vertex to list of (path_index, position) where position is 0 for start, 1 for end
    vertex_map = defaultdict(list)
    for i, path in enumerate(paths):
        if len(path) >= 2:
            vertex_map[path[0]].append((i, 0))
            vertex_map[path[-1]].append((i, 1))
    
    merged = True
    while merged:
        merged = False
        for v in list(vertex_map.keys()):
            if len(vertex_map[v]) < 2:
                continue
            
            # Find two paths that can be merged at vertex v
            entries = vertex_map[v]
            # We need one path ending at v and one starting at v (or both ending, or both starting - but only one direction works for simple merge)
            # Actually, we can merge if:
            # - Path A ends at v and Path B starts at v: merge A + B[1:]
            # - Path A starts at v and Path B ends at v: merge B[1:] + A (reversed)
            # - Path A ends at v and Path B ends at v: reverse B, then merge A + reversed_B[1:]
            # - Path A starts at v and Path B starts at v: reverse A, then merge reversed_A[1:] + B
            
            # Find pairs that can be merged
            end_paths = [e for e in entries if e[1] == 1]
            start_paths = [e for e in entries if e[1] == 0]
            
            # Try to merge an end path with a start path
            if end_paths and start_paths:
                # Don't merge a path with itself
                for ep in end_paths:
                    for sp in start_paths:
                        if ep[0] != sp[0]:
                            # Merge paths[ep[0]] (ends at v) with paths[sp[0]] (starts at v)
                            path_a = paths[ep[0]]
                            path_b = paths[sp[0]]
                            # path_a ends at v, path_b starts at v
                            # New path: path_a + path_b[1:]
                            new_path = path_a + path_b[1:]
                            # Check if new path is simple (no repeated vertices)
                            if len(set(new_path)) == len(new_path):
                                # Replace both paths with the new one
                                # Remove both, add new one
                                # We need to be careful with indices
                                new_paths = []
                                for idx, p in enumerate(paths):
                                    if idx != ep[0] and idx != sp[0]:
                                        new_paths.append(p)
                                new_paths.append(new_path)
                                paths = new_paths
                                # Rebuild vertex_map
                                vertex_map = defaultdict(list)
                                for i, path in enumerate(paths):
                                    if len(path) >= 2:
                                        vertex_map[path[0]].append((i, 0))
                                        vertex_map[path[-1]].append((i, 1))
                                merged = True
                                break
                    if merged:
                        break
            
            # If no end-start merge, try end-end or start-start merges
            if not merged and len(end_paths) >= 2:
                # Merge two paths that both end at v
                i1, i2 = end_paths[0][0], end_paths[1][0]
                path_a = paths[i1]
                path_b = paths[i2]
                # Both end at v. Reverse path_b, then merge path_a + reversed_path_b[1:]
                reversed_b = path_b[::-1]
                new_path = path_a + reversed_b[1:]
                if len(set(new_path)) == len(new_path):
                    new_paths = []
                    for idx, p in enumerate(paths):
                        if idx != i1 and idx != i2:
                            new_paths.append(p)
                    new_paths.append(new_path)
                    paths = new_paths
                    vertex_map = defaultdict(list)
                    for i, path in enumerate(paths):
                        if len(path) >= 2:
                            vertex_map[path[0]].append((i, 0))
                            vertex_map[path[-1]].append((i, 1))
                    merged = True
                    break
            
            if not merged and len(start_paths) >= 2:
                # Merge two paths that both start at v
                i1, i2 = start_paths[0][0], start_paths[1][0]
                path_a = paths[i1]
                path_b = paths[i2]
                # Both start at v. Reverse path_a, then merge reversed_path_a[1:] + path_b
                reversed_a = path_a[::-1]
                new_path = reversed_a[1:] + path_b
                if len(set(new_path)) == len(new_path):
                    new_paths = []
                    for idx, p in enumerate(paths):
                        if idx != i1 and idx != i2:
                            new_paths.append(p)
                    new_paths.append(new_path)
                    paths = new_paths
                    vertex_map = defaultdict(list)
                    for i, path in enumerate(paths):
                        if len(path) >= 2:
                            vertex_map[path[0]].append((i, 0))
                            vertex_map[path[-1]].append((i, 1))
                    merged = True
                    break
    
    return paths

# EVOLVE-BLOCK-END
