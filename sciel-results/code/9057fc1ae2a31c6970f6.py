"""Rewritten program: large clique subdivisions via max clique + greedy expansion."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

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
    # Pivot: choose vertex in P_set ∪ X with max degree in P_set
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

def _find_path(adj, src, dst, blocked):
    """Find shortest path from src to dst avoiding 'blocked' vertices (interiors)."""
    if src == dst:
        return [src]
    from collections import deque
    visited = {src}
    parent = {src: None}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        for w in adj[u]:
            if w == dst:
                # Reconstruct path
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

def _try_assign_paths(adj, branches, deadline):
    """Try to assign disjoint-interior paths for all pairs in branches."""
    n_branches = len(branches)
    branch_set = set(branches)
    pairs = list(combinations(branches, 2))
    # Sort pairs by some heuristic: prefer direct edges first
    def pair_key(p):
        u, v = p
        return 0 if v in adj[u] else 1
    pairs.sort(key=pair_key)
    
    used_interiors = set()
    paths = {}
    
    for u, v in pairs:
        if time.time() > deadline:
            return None
        # Try to find path from u to v avoiding branch_set (except endpoints) and used_interiors
        blocked = branch_set - {u, v} | used_interiors
        path = _find_path(adj, u, v, blocked)
        if path is None:
            return None
        # Check path is valid
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return None
        # Mark interiors as used
        for mid in path[1:-1]:
            if mid in branch_set or mid in used_interiors:
                return None
            used_interiors.add(mid)
        paths[(u, v)] = path
    
    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # Find maximum clique
    max_c = _max_clique(adj)
    best_branches = list(max_c)
    best_paths = None
    
    # Verify the clique solution
    paths = _try_assign_paths(adj, best_branches, deadline)
    if paths is not None:
        best_paths = paths
    
    # Try to expand: add vertices one by one
    # Sort candidates by degree (higher degree first, more likely to connect)
    candidates = sorted(set(range(n)) - set(best_branches), key=lambda v: -len(adj[v]))
    
    # Greedy expansion
    current_branches = list(best_branches)
    current_paths = _try_assign_paths(adj, current_branches, deadline)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        trial_branches = current_branches + [cand]
        trial_paths = _try_assign_paths(adj, trial_branches, deadline)
        if trial_paths is not None:
            current_branches = trial_branches
            current_paths = trial_paths
            if len(current_branches) > len(best_branches):
                best_branches = list(current_branches)
                best_paths = current_paths
    
    # Randomized restarts: try different initial cliques and expansions
    import random
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random subset as starting point
        size = random.randint(3, min(8, n))
        init = random.sample(range(n), size)
        
        # Try to make it valid
        if _try_assign_paths(adj, init, deadline) is None:
            continue
        
        current = list(init)
        current_paths = _try_assign_paths(adj, current, deadline)
        
        # Greedy expansion
        remaining = sorted(set(range(n)) - set(current), key=lambda v: -len(adj[v]))
        for cand in remaining:
            if time.time() > deadline:
                break
            trial = current + [cand]
            tp = _try_assign_paths(adj, trial, deadline)
            if tp is not None:
                current = trial
                current_paths = tp
        
        if len(current) > len(best_branches):
            best_branches = list(current)
            best_paths = current_paths
    
    # Also try: start from max clique, try different expansion orders
    if time.time() < deadline:
        for trial_order_seed in range(50):
            if time.time() > deadline:
                break
            random.seed(1000 + trial_order_seed)
            current = list(max_c)
            remaining = list(set(range(n)) - set(current))
            random.shuffle(remaining)
            remaining.sort(key=lambda v: -len(adj[v]))  # Still prefer high degree
            # Actually let's just shuffle and try
            random.shuffle(remaining)
            
            current_paths = _try_assign_paths(adj, current, deadline)
            if current_paths is None:
                continue
            
            for cand in remaining:
                if time.time() > deadline:
                    break
                trial = current + [cand]
                tp = _try_assign_paths(adj, trial, deadline)
                if tp is not None:
                    current = trial
                    current_paths = tp
            
            if len(current) > len(best_branches):
                best_branches = list(current)
                best_paths = current_paths
    
    # Final: ensure we have a valid result
    if best_paths is None:
        # Fallback: use max clique
        best_branches = list(max_c)
        best_paths = _try_assign_paths(adj, best_branches, deadline)
    
    if best_paths is None:
        # Absolute fallback: two connected vertices
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
            # Should not happen
            paths_list.append([u, v])
    
    return {
        'branches': list(best_branches),
        'paths': paths_list
    }
# EVOLVE-BLOCK-END
