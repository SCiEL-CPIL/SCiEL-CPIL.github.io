"""Rewritten program: large clique subdivisions via max-flow verification and local search."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    adj = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch(adj, R, P_set, X, best):
    if not P_set and not X:
        if len(R) > len(best[0]):
            best[0] = list(R)
        return
    pivot_set = P_set | X
    if pivot_set:
        pivot = max(pivot_set, key=lambda v: len(adj[v] & P_set))
        candidates = P_set - adj[pivot]
    else:
        candidates = P_set
    
    for v in list(candidates):
        _bron_kerbosch(adj, R | {v}, P_set & adj[v], X & adj[v], best)
        P_set -= {v}
        X |= {v}

def _max_clique(adj):
    best = [[]]
    _bron_kerbosch(adj, frozenset(), frozenset(range(P['n'])), frozenset(), best)
    return best[0]

def _bfs_path(adj, src, dst, blocked):
    """Find shortest path from src to dst avoiding 'blocked' vertices (interiors)."""
    if src == dst:
        return [src]
    visited = {src}
    parent = {src: None}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        for w in adj[u]:
            if w == dst:
                path = [dst]
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            if w not in visited and w not in blocked:
                visited.add(w)
                parent[w] = u
                queue.append(w)
    return None

def _try_solve(adj, branches, deadline):
    """Try to assign disjoint-interior paths for all pairs in branches using ordered BFS."""
    n_branches = len(branches)
    branch_set = set(branches)
    pairs = list(combinations(branches, 2))
    
    # Order pairs: prefer direct edges, then by bottleneck distance
    def pair_key(p):
        u, v = p
        # Direct edge has highest priority
        if v in adj[u]:
            return (0, 0)
        # Estimate bottleneck: minimum degree of internal nodes on shortest path
        path = _bfs_path(adj, u, v, branch_set - {u, v})
        if path is None:
            return (2, 0)
        if len(path) == 2:
            return (0, 0)
        # Use minimum degree of interior nodes as bottleneck estimate
        bottleneck = min(len(adj[mid]) for mid in path[1:-1])
        return (1, -bottleneck)
    
    pairs.sort(key=pair_key)
    
    used_interiors = set()
    paths = {}
    
    for u, v in pairs:
        if time.time() > deadline:
            return None
        blocked = branch_set - {u, v} | used_interiors
        path = _bfs_path(adj, u, v, blocked)
        if path is None:
            return None
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return None
        for mid in path[1:-1]:
            if mid in branch_set or mid in used_interiors:
                return None
            used_interiors.add(mid)
        paths[(u, v)] = path
    
    return paths

def _try_solve_with_reorder(adj, branches, deadline, max_reorders=3):
    """Try multiple orderings of pair assignments."""
    import random
    branch_set = set(branches)
    pairs = list(combinations(branches, 2))
    
    # Try the natural ordering first
    result = _try_solve(adj, branches, deadline)
    if result is not None:
        return result
    
    # Try randomized orderings
    for attempt in range(max_reorders):
        if time.time() > deadline:
            return None
        random.seed(hash(tuple(sorted(branches))) + attempt)
        shuffled_pairs = pairs[:]
        random.shuffle(shuffled_pairs)
        
        used_interiors = set()
        paths = {}
        success = True
        
        for u, v in shuffled_pairs:
            if time.time() > deadline:
                return None
            blocked = branch_set - {u, v} | used_interiors
            path = _bfs_path(adj, u, v, blocked)
            if path is None:
                success = False
                break
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    success = False
                    break
            if not success:
                break
            for mid in path[1:-1]:
                if mid in branch_set or mid in used_interiors:
                    success = False
                    break
                used_interiors.add(mid)
            if success:
                paths[(u, v)] = path
        
        if success:
            return paths
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # Find maximum clique
    max_c = _max_clique(adj)
    best_branches = list(max_c)
    best_paths = _try_solve_with_reorder(adj, best_branches, deadline)
    
    if best_paths is None:
        best_paths = _try_solve(adj, best_branches, deadline)
    
    # Greedy expansion from max clique
    current_branches = list(best_branches)
    current_paths = best_paths
    
    candidates = sorted(set(range(n)) - set(current_branches), key=lambda v: -len(adj[v]))
    
    for cand in candidates:
        if time.time() > deadline:
            break
        trial_branches = current_branches + [cand]
        trial_paths = _try_solve_with_reorder(adj, trial_branches, deadline)
        if trial_paths is not None:
            current_branches = trial_branches
            current_paths = trial_paths
            if len(current_branches) > len(best_branches):
                best_branches = list(current_branches)
                best_paths = current_paths
    
    # Local search: try swapping branches to find better configurations
    import random
    random.seed(42)
    
    # Try removing one branch and adding two others
    for _ in range(100):
        if time.time() > deadline:
            break
        
        current = list(best_branches)
        if len(current) < 3:
            break
        
        # Remove a random branch
        remove_idx = random.randint(0, len(current) - 1)
        removed = current.pop(remove_idx)
        
        # Try adding two new branches
        remaining = sorted(set(range(n)) - set(current) - {removed}, key=lambda v: -len(adj[v]))
        
        found_improvement = False
        for i in range(min(10, len(remaining))):
            for j in range(i+1, min(10, len(remaining))):
                if time.time() > deadline:
                    break
                trial = current + [remaining[i], remaining[j]]
                trial_paths = _try_solve_with_reorder(adj, trial, deadline)
                if trial_paths is not None and len(trial) > len(best_branches):
                    best_branches = list(trial)
                    best_paths = trial_paths
                    found_improvement = True
                    break
            if found_improvement or time.time() > deadline:
                break
        
        if not found_improvement:
            # Restore removed branch
            current.insert(remove_idx, removed)
    
    # Randomized restarts with different initial subsets
    for restart in range(150):
        if time.time() > deadline:
            break
        
        size = random.randint(3, min(7, n))
        init = random.sample(range(n), size)
        
        if _try_solve(adj, init, deadline) is None:
            continue
        
        current = list(init)
        current_paths = _try_solve(adj, current, deadline)
        
        remaining = sorted(set(range(n)) - set(current), key=lambda v: -len(adj[v]))
        for cand in remaining:
            if time.time() > deadline:
                break
            trial = current + [cand]
            tp = _try_solve_with_reorder(adj, trial, deadline)
            if tp is not None:
                current = trial
                current_paths = tp
        
        if len(current) > len(best_branches):
            best_branches = list(current)
            best_paths = current_paths
    
    # Fallback: ensure we have a valid result
    if best_paths is None:
        best_branches = list(max_c)
        best_paths = _try_solve(adj, best_branches, deadline)
    
    if best_paths is None:
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}
    
    # Convert to required format
    paths_list = []
    for (u, v) in combinations(best_branches, 2):
        if (u, v) in best_paths:
            paths_list.append(list(best_paths[(u, v)]))
        elif (v, u) in best_paths:
            paths_list.append(list(best_paths[(v, u)]))
        else:
            paths_list.append([u, v])
    
    return {
        'branches': list(best_branches),
        'paths': paths_list
    }
# EVOLVE-BLOCK-END
