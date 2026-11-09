"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_path(adj, src, dst, blocked):
    """Find shortest path from src to dst avoiding blocked vertices (except src/dst)."""
    if src == dst:
        return [src]
    visited = {src}
    parent = {src: None}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        if u == dst:
            # Reconstruct path
            path = []
            v = dst
            while v is not None:
                path.append(v)
                v = parent[v]
            path.reverse()
            return path
        for w in adj[u]:
            if w in visited:
                continue
            if w in blocked and w != src and w != dst:
                continue
            visited.add(w)
            parent[w] = u
            queue.append(w)
    return None

def _try_branch_set(adj, branches):
    """
    Given a set of branch vertices, try to find internally disjoint paths
    for every pair of branches. Returns (success, paths_dict) or (False, None).
    """
    bset = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    used_internal = set()
    paths = {}
    
    # Sort pairs by a heuristic: pairs that are "harder" first (fewer options)
    # We'll just do them in order, but we could sort by distance
    for b1, b2 in pairs:
        blocked = bset | used_internal
        path = _bfs_path(adj, b1, b2, blocked)
        if path is None:
            return False, None
        # Mark internal vertices as used
        for v in path[1:-1]:
            if v not in bset:
                used_internal.add(v)
        paths[(b1, b2)] = path
    
    return True, paths

def _greedy_clique_subdivision(adj, n, deadline):
    """
    Greedily build a set of branch vertices where we can find internally disjoint paths.
    Start with a triangle, then try to add vertices one by one.
    """
    # First, find a triangle
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
    
    triangle = None
    for a in range(n):
        if triangle:
            break
        for b in adj[a]:
            if b <= a:
                continue
            if (a, b) not in edge_set:
                continue
            for c in adj[b]:
                if c <= b:
                    continue
                if (b, c) in edge_set and (a, c) in edge_set:
                    triangle = [a, b, c]
                    break
            if triangle:
                break
    
    if triangle is None:
        # No triangle; fall back to just an edge
        u, v = list(edge_set)[0]
        return [int(u), int(v)], {(int(u), int(v)): [int(u), int(v)]}
    
    branches = list(triangle)
    # Verify the triangle works (it should, since they're all pairwise adjacent)
    ok, paths = _try_branch_set(adj, branches)
    if not ok:
        # Shouldn't happen for a triangle
        u, v = list(edge_set)[0]
        return [int(u), int(v)], {(int(u), int(v)): [int(u), int(v)]}
    
    # Try to add vertices greedily
    candidates = list(range(n))
    candidates.sort(key=lambda x: len(adj[x]), reverse=True)  # High degree first
    
    added = True
    while added and time.time() < deadline:
        added = False
        for v in candidates:
            if v in branches:
                continue
            new_branches = branches + [v]
            ok, new_paths = _try_branch_set(adj, new_branches)
            if ok:
                branches = new_branches
                paths = new_paths
                added = True
                break
            if time.time() >= deadline:
                break
    
    # Convert to proper types
    b_list = [int(b) for b in branches]
    p_dict = {(int(a), int(b)): [int(x) for x in path] for (a, b), path in paths.items()}
    return b_list, p_dict

def _try_all_subsets(adj, n, deadline):
    """
    Try all subsets of vertices of size k, starting from high k.
    This is exponential, so we limit to small k.
    """
    # For n=24, we can try subsets of size up to ~8 or so
    best = None
    best_size = 0
    
    # Try sizes from high to low
    max_k = min(n, 12)  # Cap at 12 to avoid explosion
    for k in range(max_k, 2, -1):
        if time.time() >= deadline:
            break
        found = False
        for subset in combinations(range(n), k):
            if time.time() >= deadline:
                found = True
                break
            ok, paths = _try_branch_set(adj, list(subset))
            if ok:
                b_list = [int(b) for b in subset]
                p_dict = {(int(a), int(b)): [int(x) for x in path] for (a, b), path in paths.items()}
                return b_list, p_dict
        if found:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s out of 90s limit
    
    n = P['n']
    adj = _build_adjacency(n, P['edges'])
    
    # Strategy 1: Greedy approach
    b_list, p_dict = _greedy_clique_subdivision(adj, n, deadline)
    greedy_size = len(b_list)
    
    # Strategy 2: Try to improve by searching for larger subsets
    # If greedy found something, try to beat it
    best_b = b_list
    best_p = p_dict
    
    if time.time() < deadline - 5.0:
        # Try a more systematic search for larger cliques
        # First, find a maximal clique (approximate)
        # Use a simple greedy coloring bound or just try to find a large clique
        
        # Try all subsets of size > greedy_size, up to a limit
        target_size = greedy_size + 1
        max_search = min(n, greedy_size + 6)
        
        found_larger = False
        for k in range(max_search, target_size - 1, -1):
            if time.time() >= deadline - 2.0:
                break
            for subset in combinations(range(n), k):
                if time.time() >= deadline - 2.0:
                    found_larger = True
                    break
                ok, paths = _try_branch_set(adj, list(subset))
                if ok:
                    best_b = [int(b) for b in subset]
                    best_p = {(int(a), int(b)): [int(x) for x in path] for (a, b), path in paths.items()}
                    found_larger = True
                    break
            if found_larger:
                break
    
    # Build the result
    branches_sorted = sorted(best_b)
    branch_set = set(branches_sorted)
    
    result_paths = []
    for b1, b2 in combinations(branches_sorted, 2):
        key = (b1, b2)
        if key in best_p:
            result_paths.append(best_p[key])
        else:
            # Fallback: direct edge
            if b2 in adj[b1]:
                result_paths.append([b1, b2])
            else:
                # Should not happen, but just in case
                result_paths.append([b1, b2])
    
    return {
        'branches': [int(b) for b in branches_sorted],
        'paths': [list(map(int, p)) for p in result_paths]
    }

# EVOLVE-BLOCK-END
