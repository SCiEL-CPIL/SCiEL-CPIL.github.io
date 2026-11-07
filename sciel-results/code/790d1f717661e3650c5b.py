import time
import sys
import random
from itertools import combinations
from collections import deque
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    def get_all_paths_limited(u, v, max_len, forbidden):
        """Find all simple paths from u to v with at most max_len vertices, avoiding forbidden."""
        if u == v:
            return [[u]]
        results = []
        def dfs(cur, path, visited):
            if len(path) > max_len:
                return
            if cur == v and len(path) > 1:
                results.append(list(path))
                return
            for nb in sorted(adj[cur]):
                if nb not in visited and nb not in forbidden:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.remove(nb)
        dfs(u, [u], {u})
        return results

    def max_clique_bronk():
        """Find maximum clique using Bron-Kerbosch with pivot."""
        best = []
        def bronk(R, P_set, X):
            nonlocal best
            if time.time() > deadline - 2:
                return
            if not P_set and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            candidates = list(P_set | X)
            if not candidates:
                return
            pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
            for v in list(P_set - adj[pivot]):
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set = P_set - {v}
                X = X | {v}
                if time.time() > deadline - 2:
                    return
        bronk(set(), set(range(n)), set())
        return best

    def solve_ilp_max_independent_set(conflict_matrix, num_paths):
        """Solve maximum independent set as ILP using scipy.optimize.milp.
        conflict_matrix[i][j] = 1 if paths i and j conflict (share interior vertex).
        Returns the set of selected path indices."""
        from scipy.optimize import milp, LinearConstraint, Bounds
        c = np.ones(num_paths)
        # x_i + x_j <= 1 for all conflicting pairs
        A_rows = []
        b_lower = []
        b_upper = []
        for i in range(num_paths):
            for j in range(i + 1, num_paths):
                if conflict_matrix[i][j] > 0:
                    row = np.zeros(num_paths)
                    row[i] = 1
                    row[j] = 1
                    A_rows.append(row)
                    b_lower.append(0)
                    b_upper.append(1)
        
        if A_rows:
            A = np.array(A_rows)
            constraints = LinearConstraint(A, b_lower, b_upper)
        else:
            constraints = LinearConstraint(np.zeros((0, num_paths)), -np.inf, np.inf)
        
        bounds = Bounds(0, 1)
        integrality = np.ones(num_paths)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                selected = set()
                for i in range(num_paths):
                    if res.x[i] > 0.5:
                        selected.add(i)
                return selected
            else:
                return None
        except Exception:
            return None

    def greedy_independent_set(conflict_matrix, num_paths):
        """Greedy maximum independent set: pick vertex with fewest conflicts first."""
        remaining = set(range(num_paths))
        selected = set()
        while remaining:
            # Pick vertex with minimum degree in conflict subgraph
            best_v = min(remaining, key=lambda v: sum(conflict_matrix[v][u] for u in remaining if u != v))
            selected.add(best_v)
            # Remove all neighbors of best_v
            to_remove = {u for u in remaining if u != best_v and conflict_matrix[best_v][u] > 0}
            remaining -= to_remove
            remaining.discard(best_v)
        return selected

    def find_branches_from_selected_paths(selected_path_indices, all_paths_info):
        """Given selected path indices, determine the branch set and verify validity.
        all_paths_info[i] = (u, v, path_list)
        Returns (branches, paths_dict) or None if invalid."""
        branches_set = set()
        paths_dict = {}
        for idx in selected_path_indices:
            u, v, path = all_paths_info[idx]
            branches_set.add(u)
            branches_set.add(v)
            paths_dict[(min(u,v), max(u,v))] = path
        
        branches = sorted(branches_set)
        
        # Verify: every pair of branches must have exactly one path
        for u, v in combinations(branches, 2):
            if (u, v) not in paths_dict:
                return None
            if (v, u) in paths_dict:
                return None
        
        # Verify: path interiors are disjoint and avoid all branches
        used_interior = set()
        for (u, v), path in paths_dict.items():
            if path[0] != u and path[0] != v:
                return None
            if path[-1] != u and path[-1] != v:
                return None
            interior = set(path[1:-1])
            if interior & branches_set:
                return None
            if interior & used_interior:
                return None
            used_interior |= interior
        
        return branches, paths_dict

    # ---------- Phase 1: Maximum clique as initial solution ----------
    best_clique = max_clique_bronk()
    
    best_branches = list(best_clique)
    best_paths = {}
    for u, v in combinations(sorted(best_clique), 2):
        best_paths[(u, v)] = [u, v]
    
    best_score = len(best_clique)

    # ---------- Phase 2: ILP-based maximum independent set on path conflict graph ----------
    # Generate candidate paths for all pairs of vertices
    # For each pair (u,v), find a few candidate paths
    # Then build conflict graph and solve MIS
    
    # Strategy: enumerate all pairs, find up to 3 shortest paths for each
    # Build conflict matrix, solve MIS, extract branch set
    
    max_path_len = 4  # max vertices in a path (including endpoints)
    
    # Collect all candidate paths
    all_candidate_paths = []  # list of (u, v, path)
    pair_to_paths = {}  # (u,v) -> list of indices into all_candidate_paths
    
    for u in range(n):
        for v in range(u + 1, n):
            forbidden = set()  # initially no forbidden
            # Find paths up to max_path_len
            paths = get_all_paths_limited(u, v, max_path_len, forbidden)
            # Also try with some forbidden to get alternative paths
            if not paths:
                # Try longer paths
                paths = get_all_paths_limited(u, v, max_path_len + 1, forbidden)
            if paths:
                # Sort by length, take up to 3
                paths.sort(key=lambda p: len(p))
                for p in paths[:3]:
                    pair_to_paths.setdefault((u, v), []).append(len(all_candidate_paths))
                    all_candidate_paths.append((u, v, p))
    
    num_paths = len(all_candidate_paths)
    
    if num_paths > 0:
        # Build conflict matrix: two paths conflict if they share an interior vertex
        # OR if they use the same pair (we can only pick one path per pair)
        conflict = np.zeros((num_paths, num_paths), dtype=np.int8)
        
        # Interior vertices for each path
        path_interiors = []
        for i, (u, v, p) in enumerate(all_candidate_paths):
            interior = set(p[1:-1])
            path_interiors.append(interior)
        
        # Conflicts due to shared interior vertices
        for i in range(num_paths):
            for j in range(i + 1, num_paths):
                if path_interiors[i] & path_interiors[j]:
                    conflict[i][j] = 1
                    conflict[j][i] = 1
        
        # Conflicts due to same pair (can only use one path per pair)
        pair_indices = {}
        for i, (u, v, p) in enumerate(all_candidate_paths):
            pair_indices.setdefault((u, v), []).append(i)
        
        for pair, indices in pair_indices.items():
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    conflict[indices[a]][indices[b]] = 1
                    conflict[indices[b]][indices[a]] = 1
        
        # Additional constraint: if we select a path for pair (u,v), then u and v become branches
        # and we cannot select any path whose interior contains u or v
        # This is automatically handled by the interior check above since
        # if u is a branch, no path can have u in its interior.
        # But we need to enforce: if path (u,v) is selected, no path (x,y) can have u or v in interior.
        # This is a more complex constraint. Let's handle it differently.
        
        # Actually, let's use a different approach:
        # A valid solution is a set of paths such that:
        # 1. At most one path per pair
        # 2. All interiors are pairwise disjoint
        # 3. No interior vertex is a branch vertex (branch = endpoint of any selected path)
        
        # Constraint 3 is tricky in the MIS formulation. Let's use a two-phase approach:
        # Phase A: Solve MIS with constraints 1 and 2 only
        # Phase B: Verify constraint 3, if violated, remove problematic paths and re-solve
        
        # For now, let's just solve MIS with constraints 1 and 2
        # and verify constraint 3 in find_branches_from_selected_paths
        
        # Try ILP first
        selected = solve_ilp_max_independent_set(conflict, num_paths)
        
        if selected is None:
            # Fallback to greedy
            selected = greedy_independent_set(conflict, num_paths)
        
        # Try to extract a valid branch set
        result = find_branches_from_selected_paths(selected, all_candidate_paths)
        if result is not None:
            branches, paths_dict = result
            if len(branches) > best_score:
                best_score = len(branches)
                best_branches = branches
                best_paths = paths_dict
        
        # Also try greedy with different orderings
        random.seed(42)
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            # Random perturbation: shuffle and use greedy
            order = list(range(num_paths))
            random.shuffle(order)
            remaining = set(order)
            selected_greedy = set()
            while remaining:
                best_v = min(remaining, key=lambda v: sum(conflict[v][u] for u in remaining if u != v))
                selected_greedy.add(best_v)
                to_remove = {u for u in remaining if u != best_v and conflict[best_v][u] > 0}
                remaining -= to_remove
                remaining.discard(best_v)
            
            result = find_branches_from_selected_paths(selected_greedy, all_candidate_paths)
            if result is not None:
                branches, paths_dict = result
                if len(branches) > best_score:
                    best_score = len(branches)
                    best_branches = branches
                    best_paths = paths_dict

    # ---------- Phase 3: Local search - try to add vertices ----------
    if time.time() < deadline - 15:
        current_branches = list(best_branches)
        current_paths = dict(best_paths)
        remaining_vertices = [v for v in range(n) if v not in set(current_branches)]
        remaining_vertices.sort(key=lambda v: -len(adj[v]))
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for v in remaining_vertices[:]:
                if time.time() > deadline - 5:
                    break
                test_branches = current_branches + [v]
                test_paths = {}
                valid = True
                used_interior = set()
                b_set = set(test_branches)
                
                for u, w in combinations(sorted(test_branches), 2):
                    if w in adj[u]:
                        test_paths[(u, w)] = [u, w]
                        continue
                    # Find shortest path avoiding other branches and used interiors
                    forbidden = (b_set - {u, w}) | used_interior
                    path = get_all_paths_limited(u, w, 5, forbidden)
                    if not path:
                        path = get_all_paths_limited(u, w, 6, forbidden)
                    if not path:
                        valid = False
                        break
                    path.sort(key=lambda p: len(p))
                    chosen = path[0]
                    test_paths[(u, w)] = chosen
                    used_interior |= set(chosen[1:-1])
                
                if valid:
                    current_branches = test_branches
                    current_paths = test_paths
                    remaining_vertices.remove(v)
                    improved = True
                    best_branches = list(current_branches)
                    best_paths = dict(current_paths)
                    best_score = len(current_branches)

    # ---------- Phase 4: Swap-based local search ----------
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            swap_improved = False
            for i in range(len(best_branches)):
                if time.time() > deadline - 5:
                    break
                for v in [x for x in range(n) if x not in set(best_branches)]:
                    if time.time() > deadline - 5:
                        break
                    test_branches = best_branches[:i] + best_branches[i+1:] + [v]
                    test_branches.sort()
                    test_paths = {}
                    valid = True
                    used_interior = set()
                    b_set = set(test_branches)
                    
                    for u, w in combinations(test_branches, 2):
                        if w in adj[u]:
                            test_paths[(u, w)] = [u, w]
                            continue
                        forbidden = (b_set - {u, w}) | used_interior
                        path = get_all_paths_limited(u, w, 5, forbidden)
                        if not path:
                            path = get_all_paths_limited(u, w, 6, forbidden)
                        if not path:
                            valid = False
                            break
                        path.sort(key=lambda p: len(p))
                        chosen = path[0]
                        test_paths[(u, w)] = chosen
                        used_interior |= set(chosen[1:-1])
                    
                    if valid:
                        best_branches = test_branches
                        best_paths = test_paths
                        swap_improved = True
                        break
                if swap_improved:
                    break
            if not swap_improved:
                break

    # ---------- Build final result ----------
    final_branches = best_branches
    final_paths = best_paths

    if final_paths is None or len(final_branches) < 2:
        final_branches = list(best_clique)
        final_paths = {}
        for u, v in combinations(sorted(final_branches), 2):
            final_paths[(u, v)] = [u, v]

    branches = [int(b) for b in sorted(final_branches)]
    paths_list = []
    for u, v in combinations(sorted(final_branches), 2):
        p = final_paths.get((u, v), [u, v])
        if p[0] != u or p[-1] != v:
            p = [u, v]
        paths_list.append([int(x) for x in p])

    return {'branches': branches, 'paths': paths_list}
