"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_disjoint_paths(branches, adj, n):
    """
    Try to find pairwise internally vertex-disjoint paths between all pairs of branches.
    Returns dict with 'branches' and 'paths' on success, None on failure.
    """
    k = len(branches)
    if k < 2:
        return None
    
    # Generate all pairs
    pairs = list(combinations(range(k), 2))
    
    # For each pair, find candidate shortest paths (up to 3)
    # We need paths that are internally disjoint
    # Strategy: use BFS to find paths, avoiding vertices already used by other paths
    
    # Order pairs by difficulty (longer paths first? or shorter first?)
    # Let's try shorter first to reduce constraints early
    def path_length_key(p):
        return len(p[1][1:-1])  # number of interior vertices
    
    # Actually, let's use a different approach:
    # Use backtracking with constraint propagation
    # Assign paths one by one, choosing from candidate paths
    
    # First, for each pair, find up to 5 candidate paths (shortest first)
    candidate_paths = {}
    for (i, j) in pairs:
        u, v = branches[i], branches[j]
        # BFS to find shortest path
        from collections import deque
        queue = deque([(u, [u])])
        visited = {u}
        found_paths = []
        while queue and len(found_paths) < 5:
            node, path = queue.popleft()
            if node == v:
                found_paths.append(path)
                continue
            for neighbor in sorted(adj[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        if not found_paths:
            return None
        
        # Sort by length
        found_paths.sort(key=len)
        candidate_paths[(i, j)] = found_paths
    
    # Now use backtracking to assign paths
    # State: which pair index we're at, which vertices are used as interiors
    def backtrack(pair_idx, used_interiors, assignment):
        if pair_idx == len(pairs):
            return True
        
        (i, j) = pairs[pair_idx]
        candidates = candidate_paths[(i, j)]
        
        for path in candidates:
            # Check if interior vertices are available
            interior = set(path[1:-1])
            if interior & used_interiors:
                continue
            if any(v in branches for v in interior):
                continue
            
            # Try this path
            assignment[(i, j)] = path
            new_used = used_interiors | interior
            if backtrack(pair_idx + 1, new_used, assignment):
                return True
            
            del assignment[(i, j)]
        
        return False
    
    assignment = {}
    if not backtrack(0, set(), assignment):
        return None
    
    # Build result
    paths_list = []
    for (i, j) in pairs:
        paths_list.append(assignment[(i, j)])
    
    return {'branches': list(branches), 'paths': [list(p) for p in paths_list]}

def _find_clique(adj, n):
    """Find the maximum clique using Bron-Kerbosch."""
    # Use greedy initialization
    best_clique = []
    
    # Start with highest degree vertex
    vertices = list(range(n))
    
    def bronk(R, P, X):
        nonlocal best_clique
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Choose pivot
        if P:
            u = max(P, key=lambda v: len(adj[v] & P) + len(adj[v] & X))
            candidates = [v for v in P if v not in adj[u]]
        else:
            candidates = list(P)
        
        for v in candidates:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
    
    bronk(set(), set(vertices), set())
    return best_clique

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_k = 0
    
    # Strategy 1: Find large cliques first
    # Try to find cliques of size 5, 6, 7, ...
    # Use Bron-Kerbosch with time limit
    
    # First, find a good initial clique
    clique = _find_clique(adj, n)
    if len(clique) >= 2:
        result = _find_disjoint_paths(clique, adj, n)
        if result is not None:
            best_result = result
            best_k = len(clique)
    
    # Strategy 2: Try to expand clique by adding vertices
    # For each vertex not in clique, try adding it
    vertices_by_degree = sorted(range(n), key=lambda v: -len(adj[v]))
    
    # Try adding vertices to the current clique
    current_branches = list(best_result['branches']) if best_result else []
    
    for candidate_vertex in vertices_by_degree:
        if candidate_vertex in current_branches:
            continue
        if time.time() > deadline:
            break
        
        new_branches = sorted(current_branches + [candidate_vertex])
        result = _find_disjoint_paths(new_branches, adj, n)
        if result is not None:
            best_result = result
            best_k = len(new_branches)
            current_branches = new_branches
    
    # Strategy 3: Local search - swap vertices
    for _ in range(100):
        if time.time() > deadline:
            break
        
        if best_result is None:
            break
        
        current_branches = list(best_result['branches'])
        improved = False
        
        # Try removing one vertex and adding another
        for remove_idx in range(len(current_branches)):
            for add_vertex in vertices_by_degree[:20]:
                if add_vertex in current_branches:
                    continue
                
                new_branches = sorted(current_branches[:remove_idx] + current_branches[remove_idx+1:] + [add_vertex])
                if len(new_branches) != len(current_branches):
                    continue
                
                result = _find_disjoint_paths(new_branches, adj, n)
                if result is not None:
                    # Check if this is better (more branches or same with different paths)
                    if len(new_branches) > len(current_branches):
                        best_result = result
                        best_k = len(new_branches)
                        current_branches = new_branches
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    # Strategy 4: Try random subsets of high-degree vertices
    if time.time() < deadline - 5:
        random.seed(42)
        for k in range(max(best_k + 1, 4), min(n, 8)):
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Pick k random vertices from top 15
                top_vertices = vertices_by_degree[:15]
                if len(top_vertices) < k:
                    break
                subset = random.sample(top_vertices, k)
                subset = sorted(subset)
                
                result = _find_disjoint_paths(subset, adj, n)
                if result is not None:
                    best_result = result
                    best_k = k
                    break
    
    # Fallback: find any edge
    if best_result is None:
        u, v = P['edges'][0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    # Ensure types are correct
    best_result['branches'] = list(best_result['branches'])
    best_result['paths'] = [list(p) for p in best_result['paths']]
    
    return best_result

# EVOLVE-BLOCK-END
