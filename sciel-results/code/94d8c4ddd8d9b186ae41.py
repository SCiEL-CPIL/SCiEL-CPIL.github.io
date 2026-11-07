"""Rewritten program using max-flow for clique subdivision search."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)

    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # --- Helper: Check if a set of branches admits a valid subdivision using max-flow ---
    def check_subdivision(branches):
        """
        Given a list of branch vertices, check if there exist vertex-disjoint paths
        between every pair of branches (interiors disjoint, avoiding other branches).
        Returns (True, paths_dict) or (False, None).
        """
        k = len(branches)
        if k < 2:
            return False, None
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # We need to find vertex-disjoint paths for all C(k,2) pairs.
        # Use a flow network with vertex splitting.
        # Each non-branch vertex has capacity 1.
        # Each branch vertex has infinite capacity (it can be used as endpoint multiple times).
        # For each pair (u,v), we need 1 unit of flow from u to v.
        # Total flow needed = C(k,2).
        
        # Build flow network:
        # Node indices: for vertex i, create i_in = 2*i, i_out = 2*i+1
        # Edge i_in -> i_out with capacity 1 if i not in branch_set, else inf
        # For each graph edge (u,v): u_out -> v_in with capacity 1, v_out -> u_in with capacity 1
        # Source: for each branch, we can send flow out. Sink: for each branch, receive flow.
        # Actually, we need to route flow from each branch to each other branch.
        # This is a multi-commodity flow problem, which is NP-hard in general.
        # However, for small k (up to ~8), we can use a different approach.
        
        # Alternative: Use the fact that for k branches, we need C(k,2) vertex-disjoint paths.
        # This is equivalent to: in the graph with branch vertices removed (and their edges),
        # can we find C(k,2) vertex-disjoint paths connecting the appropriate pairs?
        # This is still complex.
        
        # Practical approach: Use DFS with backtracking but with better pruning.
        # Since n=24 and k is small (we'll try k up to 10), this should be feasible.
        
        # Let's use a smarter DFS:
        # Order pairs by shortest path length (shortest first, easier to satisfy)
        # Use BFS to find candidate paths, try them in order.
        
        # Precompute distances from each branch
        dist_from_branch = {}
        for b in branches:
            dist_from_branch[b] = [-1] * n
            dist_from_branch[b][b] = 0
            q = deque([b])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist_from_branch[b][v] == -1:
                        dist_from_branch[b][v] = dist_from_branch[b][u] + 1
                        q.append(v)
        
        # Check connectivity
        for u, v in pairs:
            if dist_from_branch[u][v] == -1:
                return False, None
        
        # Sort pairs by shortest path distance
        pairs_sorted = sorted(pairs, key=lambda p: dist_from_branch[p[0]][p[1]])
        
        used_internal = set()
        paths_found = {}
        
        def dfs_pair(idx):
            if time.time() > deadline:
                return False
            if idx == len(pairs_sorted):
                return True
            
            u, v = pairs_sorted[idx]
            
            # If adjacent, use direct edge
            if v in adj_set[u]:
                paths_found[(u, v)] = [u, v]
                if dfs_pair(idx + 1):
                    return True
                del paths_found[(u, v)]
                return False
            
            # Find all simple paths from u to v (with length limit)
            # Using DFS with backtracking, avoiding used_internal and branches (except endpoints)
            max_path_len = dist_from_branch[u][v] + 6  # Allow some slack
            
            def find_paths(src, dst, visited, current_path, results, max_len):
                if len(results) >= 50:  # Limit number of paths to explore
                    return
                if len(current_path) > max_len:
                    return
                if src == dst:
                    results.append(list(current_path))
                    return
                for nb in adj[src]:
                    if nb in visited:
                        continue
                    if nb in used_internal:
                        continue
                    if nb in branch_set and nb != dst:
                        continue
                    visited.add(nb)
                    current_path.append(nb)
                    find_paths(nb, dst, visited, current_path, results, max_len)
                    current_path.pop()
                    visited.remove(nb)
            
            # Find candidate paths
            candidate_paths = []
            find_paths(u, v, {u}, [u], candidate_paths, max_path_len)
            
            # Sort by length (shorter first)
            candidate_paths.sort(key=len)
            
            for path in candidate_paths:
                internals = set(path[1:-1])
                if internals & used_internal:
                    continue
                if internals & branch_set:
                    continue
                used_internal.update(internals)
                paths_found[(u, v)] = path
                if dfs_pair(idx + 1):
                    return True
                used_internal.difference_update(internals)
                del paths_found[(u, v)]
            
            return False
        
        if dfs_pair(0):
            # Convert paths_found to list format
            paths_list = []
            for u, v in pairs:
                key = (u, v) if (u, v) in paths_found else (v, u)
                paths_list.append(paths_found[key])
            return True, paths_list
        else:
            return False, None

    # --- Strategy 1: Find maximum clique (fast, valid subdivision) ---
    def bronk(R, P_set, X_set):
        if not P_set and not X_set:
            return R
        max_r = None
        if P_set or X_set:
            pivot = max(P_set.union(X_set), key=lambda v: len(adj_set[v].intersection(P_set)))
            candidates = [v for v in P_set if v not in adj_set[pivot]]
        else:
            candidates = list(P_set)
        
        for v in candidates:
            R_new = R | {v}
            P_new = P_set & adj_set[v]
            X_new = X_set & adj_set[v]
            res = bronk(R_new, P_new, X_new)
            if res is not None:
                if max_r is None or len(res) > len(max_r):
                    max_r = res
            P_set.discard(v)
            X_set.add(v)
        return max_r

    best_result = None
    best_size = 0

    # Try maximum clique
    max_clique = bronk(set(), set(range(n)), set())
    if max_clique and len(max_clique) >= 2:
        branches = sorted(list(max_clique))
        paths = []
        for u, v in combinations(branches, 2):
            paths.append([u, v])
        best_result = {'branches': branches, 'paths': paths}
        best_size = len(branches)

    # --- Strategy 2: Greedy incremental branch addition ---
    # Start with a clique, then try to add vertices one by one
    
    if max_clique:
        current_branches = sorted(list(max_clique))
        # Try to add each non-branch vertex
        for candidate in range(n):
            if time.time() > deadline:
                break
            if candidate in current_branches:
                continue
            new_branches = current_branches + [candidate]
            new_branches.sort()
            ok, paths = check_subdivision(new_branches)
            if ok:
                current_branches = new_branches
                best_result = {'branches': current_branches, 'paths': paths}
                best_size = len(current_branches)
    
    # --- Strategy 3: Try random subsets of increasing size ---
    # For sizes from best_size+1 up to min(n, 12)
    for target_size in range(best_size + 1, min(n, 13)):
        if time.time() > deadline:
            break
        found = False
        # Try multiple random subsets
        for attempt in range(200):
            if time.time() > deadline:
                break
            subset = random.sample(range(n), target_size)
            subset.sort()
            ok, paths = check_subdivision(subset)
            if ok:
                best_result = {'branches': subset, 'paths': paths}
                best_size = target_size
                found = True
                break
        if found:
            continue  # Try larger size
        else:
            break  # Couldn't find this size, stop

    # --- Strategy 4: Local search / hill climbing ---
    # Start from best_result, try swapping vertices
    if best_result:
        current = list(best_result['branches'])
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for candidate in range(n):
                    if candidate in current:
                        continue
                    new_branches = current[:i] + [candidate] + current[i+1:]
                    new_branches.sort()
                    ok, paths = check_subdivision(new_branches)
                    if ok:
                        current = new_branches
                        best_result = {'branches': current, 'paths': paths}
                        improved = True
                        break
                if improved:
                    break

    # --- Fallback: Ensure we have at least 2 branches ---
    if best_result is None or len(best_result['branches']) < 2:
        # Find any edge
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}

    # Final validation and type conversion
    branches = [int(b) for b in best_result['branches']]
    paths = [list(p) for p in best_result['paths']]
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
