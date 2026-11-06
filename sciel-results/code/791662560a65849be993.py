"""Path decomposition of graphs using greedy + randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    best_results = None
    best_score = float('inf')
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = _instance(*spec)
        result = _decompose(edges, deadline)
        if result is not None:
            if best_results is None:
                best_results = []
            best_results.append(result)
            best_score = len(best_results)  # track count
    
    if best_results is None:
        # Fallback: each edge as its own path
        best_results = []
        for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
            edges = _instance(*spec)
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
    
    for attempt in range(200):
        if time.time() > deadline - 2:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        paths = _greedy_decompose(edges, rng)
        
        if paths is not None:
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                # Early stop if we hit theoretical lower bound
                lower_bound = _lower_bound(edges)
                if count <= lower_bound:
                    break
    
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


def _greedy_decompose(edges, rng):
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
        if time.time() > deadline_global:
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
        path = _find_long_path(start, adj, degree, rng, remaining)
        
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


def _find_long_path(start, adj, degree, rng, remaining):
    """
    Find a long simple path starting from start using DFS with backtracking.
    Uses randomized exploration to find long paths.
    """
    # Try to find a long path using iterative deepening or greedy with backtracking
    best_path = [start]
    
    # Use a modified DFS that tries to extend the path as far as possible
    # with randomization to escape local optima
    
    path = [start]
    visited = {start}
    
    def dfs(current, path, visited):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = path[:]
        
        # Try to extend
        neighbors = list(adj[current])
        rng.shuffle(neighbors)
        
        for nb in neighbors:
            if nb not in visited:
                path.append(nb)
                visited.add(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.discard(nb)
    
    # Limit search to avoid exponential blowup
    # Use a greedy approach with limited backtracking
    _greedy_extend(start, adj, visited, best_path, rng, max_depth=50)
    
    return best_path if len(best_path) >= 2 else None


def _greedy_extend(start, adj, visited, best_path, rng, max_depth=50):
    """
    Greedy path extension with occasional backtracking.
    """
    path = [start]
    visited = {start}
    
    for step in range(max_depth):
        current = path[-1]
        neighbors = [nb for nb in adj[current] if nb not in visited]
        
        if not neighbors:
            break
        
        # Choose next vertex: prefer those with fewer unvisited neighbors (dead-end heuristic)
        # but add randomness
        def score(nb):
            unvisited_count = sum(1 for x in adj[nb] if x not in visited)
            return unvisited_count
        
        # Sometimes pick greedily, sometimes randomly
        if rng.random() < 0.7:
            # Pick the neighbor with fewest unvisited neighbors (Warnsdorff-like)
            neighbors.sort(key=score)
            # Pick from top 3 with some randomness
            top_k = min(3, len(neighbors))
            choice = rng.choice(neighbors[:top_k])
        else:
            choice = rng.choice(neighbors)
        
        path.append(choice)
        visited.add(choice)
        
        if len(path) > len(best_path):
            best_path[:] = path[:]
    
    return path


# Global deadline reference for helper functions
deadline_global = time.time() + 145

# EVOLVE-BLOCK-END
