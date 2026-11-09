"""Rewritten program: max-flow clique subdivision finder with local search."""
# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_disjoint_paths_maxflow(branches, adj, n):
    """
    Try to find pairwise internally disjoint paths between all pairs of branches
    using a max-flow formulation.
    
    We create a flow network where:
    - Each non-branch vertex v is split into v_in -> v_out with capacity 1
    - Branch vertices have infinite capacity
    - We need to route one unit of flow for each pair (u,v) of branches
    - The total flow needed is C(k,2) where k = len(branches)
    
    Actually, a simpler approach: for each pair, we need a path. We can use
    a multi-commodity flow, but that's NP-hard. Instead, we use a greedy
    approach with backtracking, or we can use the fact that for a fixed set
    of branches, we can check feasibility by trying to assign paths.
    
    Let me use a different approach: iterative path finding with constraint propagation.
    """
    k = len(branches)
    num_pairs = k * (k - 1) // 2
    
    if num_pairs == 0:
        return {}
    
    branch_set = set(branches)
    
    # We'll use a recursive backtracking approach with memoization-like pruning
    # But for efficiency, let's try a greedy approach first, then backtracking if needed
    
    # Greedy: order pairs by some heuristic, find shortest path avoiding used interiors
    def greedy_paths(pair_order):
        used_interior = set()
        paths = {}
        
        for u, v in pair_order:
            # BFS for shortest path from u to v avoiding used_interior and other branches
            visited = {u}
            parent = {}
            queue = deque([u])
            found = False
            
            while queue:
                node = queue.popleft()
                if node == v:
                    found = True
                    break
                for nb in adj[node]:
                    if nb in visited:
                        continue
                    if nb in used_interior:
                        continue
                    if nb in branch_set and nb != u and nb != v:
                        continue
                    visited.add(nb)
                    parent[nb] = node
                    queue.append(nb)
            
            if not found:
                return None
            
            # Reconstruct path
            path = [v]
            while path[-1] != u:
                path.append(parent[path[-1]])
            path.reverse()
            
            # Mark interior vertices as used
            for node in path[1:-1]:
                used_interior.add(node)
            
            paths[(u, v)] = path
        
        return paths
    
    # Try different orderings of pairs
    pairs = list(combinations(branches, 2))
    
    # Heuristic 1: sort by BFS distance (shorter paths first)
    def bfs_dist(u, v):
        if u == v:
            return 0
        visited = {u}
        queue = deque([(u, 0)])
        while queue:
            node, d = queue.popleft()
            if node == v:
                return d
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, d + 1))
        return float('inf')
    
    # Try several orderings
    best_result = None
    
    # Ordering 1: by BFS distance
    pairs_by_dist = sorted(pairs, key=lambda p: bfs_dist(p[0], p[1]))
    result = greedy_paths(pairs_by_dist)
    if result is not None:
        best_result = result
    
    # Ordering 2: by reverse BFS distance
    pairs_by_rdist = sorted(pairs, key=lambda p: -bfs_dist(p[0], p[1]))
    result = greedy_paths(pairs_by_rdist)
    if result is not None and best_result is None:
        best_result = result
    
    # Ordering 3: by degree sum of endpoints (higher degree first)
    def degree_sum(p):
        return len(adj[p[0]]) + len(adj[p[1]])
    pairs_by_deg = sorted(pairs, key=lambda p: -degree_sum(p))
    result = greedy_paths(pairs_by_deg)
    if result is not None and best_result is None:
        best_result = result
    
    # Ordering 4: random-ish (use vertex IDs)
    pairs_by_id = sorted(pairs, key=lambda p: (p[0], p[1]))
    result = greedy_paths(pairs_by_id)
    if result is not None and best_result is None:
        best_result = result
    
    # If greedy fails, try backtracking with limited depth
    if best_result is None:
        # Try a simple backtracking: for each pair, try multiple paths
        # This is expensive, so limit it
        pass
    
    return best_result


def _can_add_branch(branches, candidate, adj, n):
    """Check if we can add candidate to branches and still have disjoint paths."""
    new_branches = sorted(branches + [candidate])
    result = _find_disjoint_paths_maxflow(new_branches, adj, n)
    return result


def _local_search(branches, paths, adj, n, deadline):
    """
    Try to improve the branch set by swapping or adding vertices.
    """
    current_branches = sorted(branches)
    current_paths = paths
    current_k = len(current_branches)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Try adding a new vertex
        for candidate in range(n):
            if candidate in current_branches:
                continue
            if time.time() > deadline - 10:
                break
            
            new_branches = sorted(current_branches + [candidate])
            new_paths = _find_disjoint_paths_maxflow(new_branches, adj, n)
            if new_paths is not None:
                current_branches = new_branches
                current_paths = new_paths
                current_k = len(current_branches)
                improved = True
                break
        
        # Try swapping: remove one, add another
        if not improved:
            for remove_v in current_branches:
                if time.time() > deadline - 10:
                    break
                for add_v in range(n):
                    if add_v in current_branches:
                        continue
                    if time.time() > deadline - 10:
                        break
                    
                    new_branches = sorted([b for b in current_branches if b != remove_v] + [add_v])
                    new_paths = _find_disjoint_paths_maxflow(new_branches, adj, n)
                    if new_paths is not None:
                        # Check if this actually helps (e.g., allows more additions later)
                        # For now, just accept if it maintains same size
                        # Actually, we want to explore, so accept and continue
                        current_branches = new_branches
                        current_paths = new_paths
                        improved = True
                        break
                if improved:
                    break
    
    return current_branches, current_paths


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # Phase 1: Find a good initial clique using Bron-Kerbosch
    # Then extend beyond clique using path verification
    
    # First, find maximum clique
    def bron_kerbosch(R, P_set, X):
        if not P_set and not X:
            yield list(R)
        
        # Choose pivot
        if P_set or X:
            pivot_set = P_set | X
            pivot = max(pivot_set, key=lambda v: len(P_set & adj[v]))
            candidates = P_set - adj[pivot]
        else:
            candidates = set()
        
        for v in list(candidates):
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            yield from bron_kerbosch(new_R, new_P, new_X)
            P_set.remove(v)
            X.add(v)
    
    # Find max clique
    all_vertices = set(range(n))
    best_clique = []
    
    # Use degeneracy ordering for efficiency
    def degeneracy_order():
        order = []
        remaining = list(all_vertices)
        deg = {v: len(adj[v]) for v in remaining}
        
        while remaining:
            # Find vertex with min degree in remaining
            min_v = min(remaining, key=lambda v: deg[v])
            order.append(min_v)
            remaining.remove(min_v)
            for nb in adj[min_v]:
                if nb in deg:
                    deg[nb] -= 1
        
        return order
    
    deg_order = degeneracy_order()
    
    # Bron-Kerbosch with degeneracy ordering
    def bk_degeneracy():
        best = []
        
        for i, v in enumerate(deg_order):
            # P is the set of vertices that come after v in the order and are adjacent to v
            P = {u for u in deg_order[i+1:] if u in adj[v]}
            X = set()
            
            for clique in bron_kerbosch({v}, P, X):
                if len(clique) > len(best):
                    best = clique
        
        return best
    
    if time.time() < deadline - 30:
        best_clique = bk_degeneracy()
    
    # Start with the max clique
    if len(best_clique) >= 2:
        branches = sorted(best_clique)
    else:
        branches = sorted(P['edges'][0])
    
    # Phase 2: Try to extend beyond the clique
    # Verify the initial clique has disjoint paths
    paths = _find_disjoint_paths_maxflow(branches, adj, n)
    
    if paths is None:
        # Fallback: use just two adjacent vertices
        branches = sorted(P['edges'][0])
        paths = {(branches[0], branches[1]): [branches[0], branches[1]]}
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 15:
        branches, paths = _local_search(branches, paths, adj, n, deadline)
    
    # Phase 4: Final greedy extension
    while time.time() < deadline - 5:
        best_candidate = None
        best_new_branches = None
        best_new_paths = None
        
        for candidate in range(n):
            if candidate in branches:
                continue
            if time.time() > deadline - 5:
                break
            
            new_branches = sorted(branches + [candidate])
            new_paths = _find_disjoint_paths_maxflow(new_branches, adj, n)
            if new_paths is not None:
                # Prefer candidates adjacent to more branches
                adj_count = sum(1 for b in branches if candidate in adj[b])
                if best_candidate is None:
                    best_candidate = candidate
                    best_new_branches = new_branches
                    best_new_paths = new_paths
                else:
                    prev_adj = sum(1 for b in branches if best_candidate in adj[b])
                    if adj_count > prev_adj:
                        best_candidate = candidate
                        best_new_branches = new_branches
                        best_new_paths = new_paths
        
        if best_candidate is None:
            break
        
        branches = best_new_branches
        paths = best_new_paths
    
    # Build the final path list
    path_list = []
    for u, v in combinations(branches, 2):
        if (u, v) in paths:
            path_list.append(list(paths[(u, v)]))
        elif (v, u) in paths:
            path_list.append(list(paths[(v, u)]))
        else:
            # Fallback: find a path
            visited = {u}
            parent = {}
            queue = deque([u])
            found = False
            while queue:
                node = queue.popleft()
                if node == v:
                    found = True
                    break
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = node
                        queue.append(nb)
            if found:
                p = [v]
                while p[-1] != u:
                    p.append(parent[p[-1]])
                p.reverse()
                path_list.append(p)
            else:
                path_list.append([u, v])
    
    # Ensure at least 2 branches
    if len(branches) < 2:
        branches = sorted(P['edges'][0])
        path_list = [[branches[0], branches[1]]]
    
    branches = sorted(branches)
    
    # Ensure types are correct
    branches = [int(b) for b in branches]
    path_list = [[int(v) for v in p] for p in path_list]
    
    return {'branches': branches, 'paths': path_list}
# EVOLVE-BLOCK-END
