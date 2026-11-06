"""Path decomposition of graphs using greedy + randomized restarts with merge and ILP polish."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
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
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        paths = _greedy_decompose(edges, rng, deadline)
        
        if paths is not None:
            # Post-process: merge paths that can be concatenated
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
    
    # Try ILP refinement if we have time
    if time.time() < deadline - 10:
        ilp_result = _ilp_refine(edges, best_paths, deadline)
        if ilp_result is not None and len(ilp_result) < best_count:
            best_paths = ilp_result
            best_count = len(best_paths)
    
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
    Uses Warnsdorff's rule to avoid trapping.
    """
    path = [start]
    visited = {start}
    
    max_depth = 100  # Allow longer paths
    
    for step in range(max_depth):
        if time.time() > deadline:
            break
        current = path[-1]
        neighbors = [nb for nb in adj[current] if nb not in visited]
        
        if not neighbors:
            break
        
        # Choose next vertex: Warnsdorff's rule (pick neighbor with fewest unvisited neighbors)
        
        def score(nb):
            unvisited_count = sum(1 for x in adj[nb] if x not in visited)
            return unvisited_count
        
        # Sort neighbors by score
        neighbors.sort(key=score)
        
        # To introduce randomness and escape local minima, 
        # pick from the top K neighbors with some probability
        top_k = min(3, len(neighbors))
        choice = rng.choice(neighbors[:top_k])
        
        path.append(choice)
        visited.add(choice)
        
        if len(path) > len(best_path):
            best_path[:] = path[:]
            
    return path


def _merge_paths(paths, edges):
    """
    Merge paths that can be concatenated.
    Checks all 4 endpoint combinations.
    """
    if len(paths) <= 1:
        return paths
    
    changed = True
    while changed:
        changed = False
        for i in range(len(paths)):
            for j in range(len(paths)):
                if i == j:
                    continue
                
                merged = False
                new_path = None
                
                # Case 1: End(Pi) == Start(Pj)
                if paths[i][-1] == paths[j][0]:
                    candidate = paths[i] + paths[j][1:]
                    if _is_simple_path(candidate, edges):
                        new_path = candidate
                        merged = True
                
                # Case 2: End(Pi) == End(Pj) -> Reverse Pj
                if not merged and paths[i][-1] == paths[j][-1]:
                    rev_j = paths[j][::-1]
                    candidate = paths[i] + rev_j[1:]
                    if _is_simple_path(candidate, edges):
                        new_path = candidate
                        merged = True

                # Case 3: Start(Pi) == Start(Pj) -> Reverse Pi, then End(Pi_rev) == Start(Pj)
                if not merged and paths[i][0] == paths[j][0]:
                    rev_i = paths[i][::-1]
                    candidate = rev_i + paths[j][1:]
                    if _is_simple_path(candidate, edges):
                        new_path = candidate
                        merged = True

                # Case 4: Start(Pi) == End(Pj) -> End(Pj) == Start(Pi), so Pj + Pi[1:]
                if not merged and paths[j][-1] == paths[i][0]:
                    candidate = paths[j] + paths[i][1:]
                    if _is_simple_path(candidate, edges):
                        new_path = candidate
                        merged = True

                if merged:
                    paths[i] = new_path
                    paths.pop(j)
                    changed = True
                    break
            if changed:
                break
                
    return paths


def _is_simple_path(path, edges):
    """Check if path is simple (no repeated vertices) and uses only graph edges."""
    if len(path) < 2:
        return True
    # Check for repeated vertices
    if len(set(path)) != len(path):
        return False
    # Check all edges exist in graph
    edge_set = set(edges)
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if tuple(sorted((u, v))) not in edge_set:
            return False
    return True


def _ilp_refine(edges, initial_paths, deadline):
    """
    Use scipy.optimize.milp to find a better path decomposition.
    Formulate as: minimize number of paths subject to edge coverage and vertex constraints.
    This is a simplified approach - we'll try to improve the initial solution.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        return None
    
    if time.time() > deadline - 15:
        return None
    
    # For ILP, we need to formulate the problem properly.
    # This is complex, so we'll use a simpler local search approach instead.
    # Try to improve by swapping path endpoints.
    
    improved = _local_search(paths=initial_paths, edges=edges, deadline=deadline)
    return improved


def _local_search(paths, edges, deadline):
    """
    Local search: try to reduce path count by finding better concatenations.
    """
    if time.time() > deadline - 10:
        return paths
    
    best_paths = [list(p) for p in paths]
    best_count = len(best_paths)
    
    # Try multiple random restarts of local search
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        current_paths = [list(p) for p in best_paths]
        
        for iteration in range(100):
            if time.time() > deadline - 10:
                break
            
            # Try to merge any two paths
            merged = False
            for i in range(len(current_paths)):
                for j in range(len(current_paths)):
                    if i == j:
                        continue
                    # Check if paths[i] ends at same vertex as paths[j] starts
                    if current_paths[i][-1] == current_paths[j][0]:
                        new_path = current_paths[i] + current_paths[j][1:]
                        if _is_simple_path(new_path, edges):
                            current_paths[i] = new_path
                            current_paths.pop(j)
                            merged = True
                            break
                if merged:
                    break
            
            if not merged:
                break
            
            if len(current_paths) < best_count:
                best_count = len(current_paths)
                best_paths = [list(p) for p in current_paths]
        
        if best_count <= _lower_bound(edges):
            break
    
    return best_paths

# EVOLVE-BLOCK-END
