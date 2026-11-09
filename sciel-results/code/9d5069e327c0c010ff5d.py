"""Clique subdivision finder: clique seed + greedy growth with max-flow verification + local search."""
# EVOLVE-BLOCK-START
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


def _max_clique(adj, deadline):
    """Bron-Kerbosch with pivot to find a maximum clique."""
    n = P['n']
    best = []

    def bron_kerbosch(R, P_set, X):
        nonlocal best
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best):
                best = list(R)
            return
        if len(R) + len(P_set) <= len(best):
            return
        PX = P_set | X
        if PX:
            pivot = max(PX, key=lambda v: len(adj[v] & P_set))
            candidates = list(P_set - adj[pivot])
        else:
            candidates = list(P_set)
        for v in candidates:
            if time.time() > deadline:
                return
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_set.discard(v)
            X.add(v)

    bron_kerbosch(set(), set(range(n)), set())
    return best


def _max_flow_disjoint_paths(adj, branches, deadline):
    """
    Verify that for each pair of branch vertices, there exist internally
    vertex-disjoint paths connecting all pairs.
    
    Uses max-flow with vertex splitting:
    - Each vertex v is split into v_in (2v) and v_out (2v+1)
    - Non-branch vertices have capacity 1 on the split edge
    - Branch vertices have infinite capacity
    - For each pair (u,v), we need one unit of flow from u to v
    
    Since this is a multi-commodity flow (each pair needs its own flow),
    we handle it by: for each pair, check if there's a path in the
    residual capacity graph after reserving interior vertices for
    previously assigned pairs.
    
    Returns (success, paths_dict) where paths_dict maps (u,v) to path list.
    """
    n = P['n']
    k = len(branches)
    if k < 2:
        return False, {}
    
    branch_set = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    
    # Track which non-branch vertices are already used as interior
    used_interior = set()
    
    # For efficiency, precompute adjacency
    adj_list = [list(adj[v]) for v in range(n)]
    
    # Sort pairs by "difficulty" - pairs that are adjacent in the graph first
    # (easiest to route), then by shortest path length
    def pair_difficulty(pair):
        u, v = pair
        if v in adj[u]:
            return 0  # direct edge, no interior needed
        # BFS distance
        visited = {u}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == v:
                return dist
            for nb in adj_list[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return 10**9  # no path
    
    pairs_sorted = sorted(pairs, key=pair_difficulty)
    
    assignment = {}
    
    for u, v in pairs_sorted:
        if time.time() > deadline:
            return False, {}
        
        # Find a path from u to v avoiding used_interior and other branches in interior
        # Use BFS with the constraint that interior vertices must not be in used_interior
        # and must not be branches (other than u and v)
        
        # If u and v are adjacent, use the direct edge
        if v in adj[u]:
            assignment[(u, v)] = [u, v]
            continue
        
        # BFS to find shortest path avoiding used_interior and other branches
        found_path = None
        visited = {u}
        parent = {u: None}
        queue = deque([u])
        while queue:
            node = queue.popleft()
            if node == v:
                # Reconstruct path
                path = []
                cur = v
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                found_path = path
                break
            for nb in adj_list[node]:
                if nb not in visited:
                    # Interior vertices (not u, not v) must not be used_interior or branches
                    if nb != u and nb != v:
                        if nb in used_interior:
                            continue
                        if nb in branch_set:
                            continue
                    visited.add(nb)
                    parent[nb] = node
                    queue.append(nb)
        
        if found_path is None:
            return False, {}
        
        # Reserve interior vertices
        interior = set(found_path[1:-1])
        used_interior.update(interior)
        assignment[(u, v)] = found_path
    
    return True, assignment


def _verify_and_extract(adj, branches, deadline):
    """Verify branch set and extract paths. Returns (success, paths_dict)."""
    return _max_flow_disjoint_paths(adj, branches, deadline)


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = P['n']
    adj = _build_adj()
    adj_list = [list(adj[v]) for v in range(n)]
    
    # Step 1: Find maximum clique as seed
    clique = _max_clique(adj, deadline)
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    branches = sorted(clique)
    branch_set = set(branches)
    
    # Verify the clique (should always succeed since clique means all pairs adjacent)
    success, paths_dict = _verify_and_extract(adj, branches, deadline)
    if not success:
        # Fallback: just use an edge
        u, v = P['edges'][0]
        branches = [u, v]
        branch_set = set(branches)
        paths_dict = {(u, v): [u, v]}
    
    best_branches = list(branches)
    best_paths_dict = dict(paths_dict)
    
    # Step 2: Greedy growth - try adding each non-branch vertex
    # Sort candidates by degree (higher degree first)
    candidates = [v for v in range(n) if v not in branch_set]
    candidates.sort(key=lambda v: -len(adj[v]))
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for cand in candidates[:]:
            if cand in branch_set:
                continue
            if time.time() > deadline - 5.0:
                break
            
            # Try adding candidate
            new_branches = branches + [cand]
            new_branch_set = branch_set | {cand}
            
            success, new_paths = _verify_and_extract(adj, new_branches, deadline)
            if success:
                branches = new_branches
                branch_set = new_branch_set
                paths_dict = new_paths
                candidates.remove(cand)
                improved = True
                # Update best
                best_branches = list(branches)
                best_paths_dict = dict(paths_dict)
    
    # Step 3: Local search - try swap operations
    # For each branch b and non-branch c, try swapping b with c
    if time.time() < deadline - 3.0:
        non_branches = [v for v in range(n) if v not in branch_set]
        non_branches.sort(key=lambda v: -len(adj[v]))
        
        swap_improved = True
        while swap_improved and time.time() < deadline - 3.0:
            swap_improved = False
            for b in branches[:]:
                if time.time() > deadline - 3.0:
                    break
                for c in non_branches[:]:
                    if time.time() > deadline - 3.0:
                        break
                    # Try swapping b with c
                    new_branches = [v for v in branches if v != b] + [c]
                    new_branch_set = (branch_set - {b}) | {c}
                    
                    success, new_paths = _verify_and_extract(adj, new_branches, deadline)
                    if success:
                        # Check if this is an improvement (it's same size, but might enable further growth)
                        branches = new_branches
                        branch_set = new_branch_set
                        paths_dict = new_paths
                        non_branches = [v for v in range(n) if v not in branch_set]
                        non_branches.sort(key=lambda v: -len(adj[v]))
                        swap_improved = True
                        break
                if swap_improved:
                    break
    
    # Step 4: Try to grow again after local search
    if time.time() < deadline - 2.0:
        candidates = [v for v in range(n) if v not in branch_set]
        candidates.sort(key=lambda v: -len(adj[v]))
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for cand in candidates[:]:
                if cand in branch_set:
                    continue
                if time.time() > deadline - 2.0:
                    break
                
                new_branches = branches + [cand]
                success, new_paths = _verify_and_extract(adj, new_branches, deadline)
                if success:
                    branches = new_branches
                    branch_set = branch_set | {cand}
                    paths_dict = new_paths
                    candidates.remove(cand)
                    improved = True
                    best_branches = list(branches)
                    best_paths_dict = dict(paths_dict)
    
    # Use the best found so far
    if len(best_branches) >= len(branches):
        branches = best_branches
        paths_dict = best_paths_dict
    else:
        # Final verification
        success, paths_dict = _verify_and_extract(adj, branches, deadline)
        if not success:
            branches = best_branches
            paths_dict = best_paths_dict
    
    # Build output
    paths = []
    for u, v in combinations(sorted(branches), 2):
        if (u, v) in paths_dict:
            paths.append(list(paths_dict[(u, v)]))
        elif (v, u) in paths_dict:
            paths.append(list(paths_dict[(v, u)]))
        else:
            # Fallback: direct edge if exists, else [u, v]
            if v in adj[u]:
                paths.append([u, v])
            else:
                paths.append([u, v])
    
    # Ensure types are correct
    branches = [int(b) for b in branches]
    paths = [[int(x) for x in p] for p in paths]
    
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
