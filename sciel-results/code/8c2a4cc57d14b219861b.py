"""
Maximum clique subdivision (topological minor) finder.

Approach:
1. Find maximum clique via Bron-Kerbosch (gives guaranteed valid subdivision with edge-paths).
2. Try to grow the branch set by adding vertices, verifying each time that
   internally vertex-disjoint paths exist for all pairs (via max-flow).
3. Use branch-and-bound to search for larger branch sets with valid paths.
"""
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degeneracy_order(adj):
    n = P['n']
    remaining = list(range(n))
    remaining_set = set(remaining)
    order = []
    while remaining:
        min_deg = min(len(adj[v] & remaining_set) for v in remaining)
        v = next(v for v in remaining if len(adj[v] & remaining_set) == min_deg)
        order.append(v)
        remaining.remove(v)
        remaining_set.discard(v)
    return order

def _bron_kerbosch(adj, R, P_set, X, best_clique, deadline):
    if time.time() > deadline:
        return best_clique
    if not P_set and not X:
        if len(R) > len(best_clique):
            best_clique = list(R)
        return best_clique
    if len(R) + len(P_set) <= len(best_clique):
        return best_clique
    PX = P_set | X
    if PX:
        pivot = max(PX, key=lambda v: len(adj[v] & P_set))
        candidates = list(P_set - adj[pivot])
    else:
        candidates = list(P_set)
    for v in candidates:
        if time.time() > deadline:
            return best_clique
        new_R = R | {v}
        new_P = P_set & adj[v]
        new_X = X & adj[v]
        best_clique = _bron_kerbosch(adj, new_R, new_P, new_X, best_clique, deadline)
        P_set.discard(v)
        X.add(v)
    return best_clique

def _max_clique(adj, deadline):
    n = P['n']
    order = _degeneracy_order(adj)
    best = []
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
        P_set = set(w for w in adj[v] if w in order[i+1:])
        if not P_set:
            continue
        X = set(w for w in adj[v] if w in order[:i])
        best = _bron_kerbosch(adj, {v}, P_set, X, best, deadline)
    if not best:
        best = _bron_kerbosch(adj, set(), set(range(n)), set(), best, deadline)
    return best

def _find_disjoint_paths_maxflow(adj, branches, deadline):
    """
    Use max-flow to find internally vertex-disjoint paths for all pairs of branches.
    Returns (success, paths_dict) where paths_dict maps (u,v) -> path list.
    
    Strategy: For each pair, we need a path. We model this as a flow problem
    where we send one unit of flow for each pair through a shared network
    with vertex capacities of 1 for non-branch vertices.
    """
    n = P['n']
    k = len(branches)
    if k < 2:
        return False, {}
    
    branch_set = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    num_pairs = len(pairs)
    
    # Build a flow network with vertex splitting
    # Node 2*v = v_in, 2*v+1 = v_out
    SS = 2 * n       # super source
    TT = 2 * n + 1   # super sink
    total_nodes = 2 * n + 2
    
    # We'll use a single max-flow per pair approach with residual tracking
    # Actually, let's use a different approach: iteratively find paths with
    # BFS, marking used vertices, and if stuck, try alternatives.
    
    # Better approach: Use max-flow with a super source connected to all branch
    # "in" nodes and super sink connected from all branch "out" nodes, but
    # we need to ensure each pair gets exactly one path.
    
    # Simplest correct approach: Try to find paths greedily with backtracking.
    # For small k (up to ~8), this is feasible.
    
    # First, find all simple paths between each pair (limited length)
    pair_candidate_paths = {}
    for u, v in pairs:
        paths = []
        # BFS to find shortest path first
        visited = {u}
        queue = deque([(u, [u])])
        while queue and len(paths) < 20:
            node, path = queue.popleft()
            if len(path) > 10:  # limit path length
                continue
            if node == v:
                paths.append(path)
                continue
            for nb in sorted(adj[node]):
                if nb not in visited:
                    if nb in branch_set and nb != u and nb != v:
                        continue
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        pair_candidate_paths[(u, v)] = paths
    
    # Check if all pairs have at least one path
    for u, v in pairs:
        if not pair_candidate_paths[(u, v)]:
            return False, {}
    
    # Now try to assign disjoint paths using backtracking
    # Sort pairs by number of candidates (ascending) for efficiency
    pairs_sorted = sorted(pairs, key=lambda p: len(pair_candidate_paths[p]))
    
    used_interior = set()
    
    def backtrack(idx, assignment):
        if time.time() > deadline:
            return None
        if idx == len(pairs_sorted):
            return assignment
        
        u, v = pairs_sorted[idx]
        for path in pair_candidate_paths[(u, v)]:
            interior = set(path[1:-1])
            if interior.isdisjoint(used_interior):
                used_interior.update(interior)
                result = backtrack(idx + 1, assignment + [(u, v, path)])
                if result is not None:
                    return result
                used_interior.difference_update(interior)
        
        return None
    
    result = backtrack(0, [])
    if result is None:
        return False, {}
    
    paths_dict = {}
    for u, v, path in result:
        paths_dict[(u, v)] = path
    
    return True, paths_dict

def _try_grow(adj, branches, candidates, deadline):
    """Try to add vertices to branches one at a time."""
    branch_set = set(branches)
    current_branches = list(branches)
    
    # Sort candidates by degree (higher first)
    candidates_sorted = sorted(candidates, key=lambda v: -len(adj[v]))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in candidates_sorted[:]:
            if cand in branch_set:
                continue
            if time.time() > deadline:
                break
            new_branches = current_branches + [cand]
            success, _ = _find_disjoint_paths_maxflow(adj, new_branches, deadline)
            if success:
                current_branches.append(cand)
                branch_set.add(cand)
                candidates_sorted.remove(cand)
                improved = True
    
    return current_branches

def _branch_and_bound(adj, initial_branches, all_vertices, deadline):
    """
    Branch-and-bound to find maximum branch set with valid disjoint paths.
    State: (current_branches, available_vertices)
    """
    best = list(initial_branches)
    best_size = len(best)
    
    def bbs(current, available, depth):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        
        # Upper bound: current + all available
        if len(current) + len(available) <= best_size:
            return
        
        # Try to add each available vertex
        # Sort by degree for better pruning
        for v in sorted(available, key=lambda x: -len(adj[x])):
            if time.time() > deadline:
                return
            new_current = current + [v]
            new_available = [x for x in available if x != v]
            
            success, _ = _find_disjoint_paths_maxflow(adj, new_current, deadline)
            if success:
                if len(new_current) > best_size:
                    best = list(new_current)
                    best_size = len(new_current)
                bbs(new_current, new_available, depth + 1)
    
    bbs(list(initial_branches), list(all_vertices), 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin
    
    n = P['n']
    adj = _build_adj()
    
    # Step 1: Find maximum clique
    clique = _max_clique(adj, deadline)
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    branches = sorted(clique)
    
    # Verify clique has valid paths (should be trivial - just edges)
    success, paths_dict = _find_disjoint_paths_maxflow(adj, branches, deadline)
    if not success:
        u, v = P['edges'][0]
        branches = [u, v]
        paths_dict = {(u, v): [u, v]}
    
    # Step 2: Try to grow using greedy approach
    branch_set = set(branches)
    candidates = [v for v in range(n) if v not in branch_set]
    grown = _try_grow(adj, branches, candidates, deadline)
    if len(grown) > len(branches):
        branches = sorted(grown)
        branch_set = set(branches)
    
    # Step 3: Try branch-and-bound for further improvement
    if time.time() < deadline - 10:
        remaining = [v for v in range(n) if v not in branch_set]
        # Limit B&B to avoid timeout
        bb_deadline = min(deadline, time.time() + 40)
        best_bb = _branch_and_bound(adj, branches, remaining, bb_deadline)
        if len(best_bb) > len(branches):
            branches = sorted(best_bb)
    
    # Final path extraction
    success, paths_dict = _find_disjoint_paths_maxflow(adj, branches, deadline)
    if not success:
        # Fallback: use clique
        branches = sorted(clique)
        success, paths_dict = _find_disjoint_paths_maxflow(adj, branches, deadline)
        if not success:
            u, v = P['edges'][0]
            branches = [u, v]
            paths_dict = {(u, v): [u, v]}
    
    # Build output
    paths = []
    for u, v in combinations(sorted(branches), 2):
        if (u, v) in paths_dict:
            paths.append(list(paths_dict[(u, v)]))
        elif (v, u) in paths_dict:
            paths.append(list(paths_dict[(v, u)]))
        else:
            paths.append([u, v])
    
    # Ensure types are correct
    branches = [int(b) for b in branches]
    paths = [list(int(x) for x in p) for p in paths]
    
    return {'branches': branches, 'paths': paths}
