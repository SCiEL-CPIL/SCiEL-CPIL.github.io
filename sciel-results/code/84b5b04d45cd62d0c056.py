"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)
adj_list = [sorted(adj[i]) for i in range(N)]

DEADLINE = time.time() + 85

def time_left():
    return DEADLINE - time.time()

# Precompute common neighbor bitmasks for all pairs
# mask[v] is a bitmask of adj[v]
mask = [0] * N
for v in range(N):
    for u in adj[v]:
        mask[v] |= (1 << u)

def common_neighbors(u, v):
    """Return bitmask of common neighbors of u and v."""
    return mask[u] & mask[v]

def greedy_coloring(candidates, adj_set):
    """Greedy coloring of candidates, return (num_colors, color_of)"""
    colors = {}
    for v in candidates:
        used = set()
        for u in candidates:
            if u != v and u in colors and u in adj_set[v]:
                used.add(colors[u])
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    if not colors:
        return 0, colors
    return max(colors.values()) + 1, colors

def max_clique_branch_bound(adj_set, deadline):
    """Find maximum clique using branch and bound with greedy coloring."""
    n = len(adj_set)
    best_clique = []
    
    def expand(candidates, current):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not candidates:
            if len(current) > len(best_clique):
                best_clique = list(current)
            return
        
        num_colors, color_map = greedy_coloring(candidates, adj_set)
        if len(current) + num_colors <= len(best_clique):
            return
        
        # Process candidates in reverse color order
        for i in range(len(candidates) - 1, -1, -1):
            if time.time() > deadline:
                return
            v = candidates[i]
            if len(current) + i + 1 <= len(best_clique):
                break
            new_candidates = [u for u in candidates[:i] if u in adj_set[v]]
            new_current = current + [v]
            expand(new_candidates, new_current)
            candidates = candidates[:i]
            # Rebound after removing v
            num_colors, _ = greedy_coloring(candidates, adj_set)
            if len(current) + num_colors <= len(best_clique):
                return
    
    expand(list(range(n)), [])
    return best_clique

def find_disjoint_paths(branches):
    """
    Given branch vertices, find one simple path for each unordered pair
    with pairwise disjoint interiors avoiding all branches.
    Returns list of paths or None.
    """
    n_b = len(branches)
    if n_b < 2:
        return None
    branch_set = set(branches)
    pairs = list(combinations(range(n_b), 2))
    
    # Sort pairs: direct edges first (shortest paths), then by BFS distance
    def pair_key(p):
        u, v = branches[p[0]], branches[p[1]]
        if v in adj[u]:
            return (0, 0)
        # BFS distance
        visited = {u}
        q = deque([(u, 0)])
        while q:
            curr, dist = q.popleft()
            for w in adj_list[curr]:
                if w == v:
                    return (1, dist + 1)
                if w not in visited and w not in branch_set:
                    visited.add(w)
                    q.append((w, dist + 1))
        return (1, float('inf'))
    
    pairs.sort(key=pair_key)
    
    used_interior = set()
    paths = {}
    
    for (i, j) in pairs:
        s, t = branches[i], branches[j]
        parent = {}
        visited = {s}
        q = deque([s])
        found = False
        while q:
            u = q.popleft()
            if u == t:
                found = True
                break
            for w in adj_list[u]:
                if w in visited:
                    continue
                if w in branch_set and w != t:
                    continue
                if w in used_interior:
                    continue
                visited.add(w)
                parent[w] = u
                q.append(w)
        if not found:
            return None
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        for v in path[1:-1]:
            used_interior.add(v)
        paths[(i, j)] = path
    
    result = []
    for (i, j) in combinations(range(n_b), 2):
        result.append(paths[(i, j)])
    return result

def try_branch_set(branches):
    return find_disjoint_paths(branches)

def local_search_extend(branches, deadline):
    """Try to add one more vertex to the branch set."""
    n_b = len(branches)
    for candidate in range(N):
        if candidate in branches:
            continue
        if time_left() < 2:
            break
        trial = list(branches) + [candidate]
        paths = try_branch_set(trial)
        if paths is not None:
            return trial, paths
    return None, None

def search_from_seed(seed_branches, deadline):
    """Starting from a seed, try to build up the largest routable branch set."""
    branches = list(seed_branches)
    paths = try_branch_set(branches)
    if paths is None:
        return None, None
    
    improved = True
    while improved and time_left() > 3:
        improved = False
        # Try adding vertices
        for candidate in range(N):
            if candidate in branches:
                continue
            if time_left() < 2:
                break
            trial = list(branches) + [candidate]
            trial_paths = try_branch_set(trial)
            if trial_paths is not None:
                branches = trial
                paths = trial_paths
                improved = True
                break
        # Try swaps if no addition worked
        if not improved and time_left() > 3:
            for i in range(len(branches)):
                if time_left() < 2:
                    break
                for candidate in range(N):
                    if candidate in branches:
                        continue
                    trial = list(branches)
                    trial[i] = candidate
                    trial_paths = try_branch_set(trial)
                    if trial_paths is not None:
                        # Check if this swap allows further extension
                        extended, ext_paths = local_search_extend(trial, deadline)
                        if extended is not None and len(extended) > len(branches):
                            branches = extended
                            paths = ext_paths
                            improved = True
                            break
                if improved:
                    break
    
    return branches, paths

def construct():
    deadline = DEADLINE
    best_branches = None
    best_paths = None
    
    # Step 1: Find maximum clique
    max_clique = max_clique_branch_bound(adj, deadline - 10)
    
    if len(max_clique) >= 2:
        branches, paths = search_from_seed(max_clique, deadline - 5)
        if branches is not None and paths is not None:
            best_branches = branches
            best_paths = paths
    
    # Step 2: Try multiple seeds to find larger routable sets
    # Use vertices sorted by degree as seeds
    deg_sorted = sorted(range(N), key=lambda v: -len(adj[v]))
    
    # Try pairs of high-degree vertices as seeds
    tried_seeds = set()
    for i in range(min(10, len(deg_sorted))):
        for j in range(i+1, min(10, len(deg_sorted))):
            seed = [deg_sorted[i], deg_sorted[j]]
            seed_key = tuple(sorted(seed))
            if seed_key in tried_seeds:
                continue
            tried_seeds.add(seed_key)
            
            if time_left() < 5:
                break
            
            branches, paths = search_from_seed(seed, deadline - 3)
            if branches is not None and paths is not None:
                if best_branches is None or len(branches) > len(best_branches):
                    best_branches = branches
                    best_paths = paths
        
        if time_left() < 5:
            break
    
    # Step 3: If we still have time, try more seeds
    if time_left() > 10:
        for i in range(len(deg_sorted)):
            if time_left() < 3:
                break
            for j in range(i+1, len(deg_sorted)):
                seed = [deg_sorted[i], deg_sorted[j]]
                seed_key = tuple(sorted(seed))
                if seed_key in tried_seeds:
                    continue
                tried_seeds.add(seed_key)
                
                if time_left() < 2:
                    break
                
                branches, paths = search_from_seed(seed, deadline - 1)
                if branches is not None and paths is not None:
                    if best_branches is None or len(branches) > len(best_branches):
                        best_branches = branches
                        best_paths = paths
    
    # Fallback: ensure we have at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        # Find any edge
        for u, v in P['edges']:
            if v in adj[u]:
                best_branches = [u, v]
                best_paths = [[u, v]]
                break
    
    # Build output
    if best_paths is not None:
        path_list = [list(p) for p in best_paths]
    else:
        # Should not happen, but just in case
        path_list = []
        for i in range(len(best_branches)):
            for j in range(i+1, len(best_branches)):
                u, v = best_branches[i], best_branches[j]
                # Direct edge
                path_list.append([u, v])
    
    return {'branches': [int(b) for b in best_branches], 'paths': path_list}
# EVOLVE-BLOCK-END
