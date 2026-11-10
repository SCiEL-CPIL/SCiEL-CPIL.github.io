"""Rewritten program: large clique subdivisions in a pinned graph."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

# Precompute adjacency
N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)
adj_list = [sorted(adj[i]) for i in range(N)]

DEADLINE = time.time() + 50  # safety margin before 60s (kill at 90s)

def time_left():
    return DEADLINE - time.time()

def find_disjoint_paths(branches, n_branches):
    """
    Given a list of branch vertices, find one simple path for each unordered pair
    such that all path interiors are pairwise disjoint and avoid all branches.
    Uses incremental BFS with branch-vertex removal.
    Returns dict of (i,j) -> path, or None if impossible.
    """
    if n_branches < 2:
        return None
    branch_set = set(branches)
    # We need to route paths for all C(n,2) pairs with disjoint interiors.
    # Strategy: process pairs in order, each time BFS avoiding already-used interior vertices.
    # To maximize success, process pairs where direct edge exists first (shortest paths).
    
    pairs = list(combinations(range(n_branches), 2))
    # Sort pairs: prefer pairs that have a direct edge (shortest possible path)
    def pair_key(p):
        u, v = branches[p[0]], branches[p[1]]
        return 0 if v in adj[u] else 1
    pairs.sort(key=pair_key)
    
    used_interior = set()
    paths = {}
    
    for (i, j) in pairs:
        s, t = branches[i], branches[j]
        # BFS from s to t, avoiding used_interior and all branches except s and t
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
        # Reconstruct path
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        # Mark interior vertices
        for v in path[1:-1]:
            used_interior.add(v)
        paths[(i, j)] = path
    
    # Build result in canonical order
    result = {}
    for (i, j) in combinations(range(n_branches), 2):
        result[(i, j)] = paths[(i, j)]
    return result

def try_branch_set(branches):
    """Try to route disjoint paths for the given branch set."""
    n = len(branches)
    result = find_disjoint_paths(branches, n)
    if result is None:
        return None
    # Format for return
    path_list = []
    for (i, j) in combinations(range(n), 2):
        path_list.append(result[(i, j)])
    return path_list

def greedy_build_branches(max_k, deadline_check=None):
    """
    Greedily build a branch set by adding vertices one at a time,
    checking feasibility after each addition.
    Returns (branches, paths) or None.
    """
    # Start with a promising seed: highest-degree vertices
    deg = [len(adj[i]) for i in range(N)]
    order = sorted(range(N), key=lambda x: -deg[x])
    
    branches = []
    # Try to build up to max_k
    for candidate in order:
        if len(branches) >= max_k:
            break
        if deadline_check and deadline_check() < 2:
            break
        trial = branches + [candidate]
        paths = try_branch_set(trial)
        if paths is not None:
            branches = trial
            if len(branches) >= max_k:
                break
    
    if len(branches) < 2:
        return None
    paths = try_branch_set(branches)
    if paths is None:
        return None
    return branches, paths

def local_search(branches, paths, deadline_check=None):
    """
    Try to improve by swapping one branch for another.
    """
    n = len(branches)
    current_score = n
    best = (branches, paths)
    
    # Try adding one more vertex
    for candidate in range(N):
        if candidate in branches:
            continue
        if deadline_check and deadline_check() < 2:
            break
        trial = list(branches) + [candidate]
        trial_paths = try_branch_set(trial)
        if trial_paths is not None:
            return trial, trial_paths
    
    # Try swapping
    for i in range(len(branches)):
        for candidate in range(N):
            if candidate in branches:
                continue
            if deadline_check and deadline_check() < 1:
                break
            trial = list(branches)
            trial[i] = candidate
            trial_paths = try_branch_set(trial)
            if trial_paths is not None:
                # Same size but different set; try to extend
                extended, ext_paths = local_search(trial, trial_paths, deadline_check)
                if extended is not None and len(extended) > len(branches):
                    return extended, ext_paths
    return best

def search_max_branches():
    """
    Main search: try to find the maximum number of branches.
    """
    def deadline_check():
        return time_left()
    
    best_branches = None
    best_paths = None
    best_count = 1  # minimum: 2 branches, but start with fallback
    
    # Fallback: always valid
    u, v = P['edges'][0]
    fallback = ([u, v], [[u, v]])
    
    # Try greedy from different starting points
    # First, try to find large sets
    
    # Strategy 1: Greedy from highest-degree
    result = greedy_build_branches(N, deadline_check)
    if result is not None:
        b, p = result
        if len(b) > best_count:
            best_count = len(b)
            best_branches = b
            best_paths = p
    
    # Strategy 2: Try all pairs as seeds, then extend
    if time_left() > 10:
        # Try extending from each edge
        for (u, v) in P['edges']:
            if time_left() < 5:
                break
            trial = [u, v]
            paths = try_branch_set(trial)
            if paths is None:
                continue
            # Extend greedily
            extended, ext_paths = local_search(trial, paths, deadline_check)
            if extended is not None and len(extended) > best_count:
                best_count = len(extended)
                best_branches = extended
                best_paths = ext_paths
    
    # Strategy 3: Random restarts
    import random
    if time_left() > 5:
        rng = random.Random(42)
        for _ in range(200):
            if time_left() < 3:
                break
            # Pick a random starting pair
            u = rng.randint(0, N-1)
            v = rng.choice(adj_list[u])
            trial = [u, v]
            paths = try_branch_set(trial)
            if paths is None:
                continue
            extended, ext_paths = local_search(trial, paths, deadline_check)
            if extended is not None and len(extended) > best_count:
                best_count = len(extended)
                best_branches = extended
                best_paths = ext_paths
    
    # Final: also try to extend the best found
    if best_branches is not None and time_left() > 3:
        extended, ext_paths = local_search(best_branches, best_paths, deadline_check)
        if extended is not None and len(extended) > best_count:
            best_count = len(extended)
            best_branches = extended
            best_paths = ext_paths
    
    if best_branches is None:
        return fallback
    
    return best_branches, best_paths

def construct():
    global DEADLINE
    DEADLINE = time.time() + 50
    
    branches, paths = search_max_branches()
    
    # Ensure types are correct
    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
