import time
import random
from itertools import combinations
from collections import deque

# The graph definition from the prompt
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]

    def check_time():
        return time.time() < deadline

    def find_max_clique():
        """Find maximum clique using Bron-Kerbosch with pivot."""
        best_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if not check_time():
                return
            
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            
            # Choose pivot
            if P:
                pivot = max(P, key=lambda v: len(adj[v] & (P | X)))
            else:
                pivot = None
            
            candidates = set(P)
            if pivot is not None:
                candidates -= adj[pivot]
            
            for v in list(candidates):
                if not check_time():
                    return
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return best_clique

    def find_disjoint_paths(branches, max_paths_to_try=10, recursion_limit=1000):
        """
        Try to find vertex-disjoint paths between all pairs of branches.
        Returns a dict {pair: path} or None.
        """
        if len(branches) < 2:
            return {}
            
        branches = sorted(branches)
        pairs = list(combinations(branches, 2))
        
        # We will use a backtracking approach.
        # State: used_interiors (set), remaining_pairs (list of tuples)
        
        def backtrack(remaining_pairs, used_interiors, current_paths):
            if not check_time():
                return None
            
            if not remaining_pairs:
                return current_paths
            
            # Heuristic: Pick the pair with the fewest available shortest paths?
            # For simplicity, pick the first one, but we can shuffle pairs initially.
            # A better heuristic: Pick the pair (u,v) such that u and v have the fewest 
            # available neighbors not in used_interiors.
            
            u, v = remaining_pairs[0]
            
            # Check if u and v are still connected in the available graph
            # Find all simple paths from u to v avoiding used_interiors and other branches
            # We limit the search to shortest paths or a few random ones
            
            # BFS to find shortest path length
            # Then enumerate paths up to a certain length?
            # Or just find one path, try it. If fail, find next?
            
            # Let's find the shortest path first.
            # Then try random perturbations?
            
            # Efficient approach: 
            # 1. Find one shortest path.
            # 2. If it works, great.
            # 3. If not, we need to find *another* path.
            
            # To find multiple paths, we can use a "k-shortest paths" algorithm or simple DFS with backtracking on the path selection itself.
            
            # Let's use a simple DFS to find paths from u to v.
            # We will collect up to `max_paths_to_try` distinct paths.
            
            paths_found = []
            
            # DFS for paths
            def dfs_paths(node, target, path, visited):
                if len(paths_found) >= max_paths_to_try:
                    return
                if node == target:
                    paths_found.append(list(path))
                    return
                
                # Sort neighbors by degree (greedy: try high degree first? or low degree to find bottlenecks?)
                # Let's try neighbors in a somewhat random or sorted order to get diverse paths
                neighbors = [x for x in adj[node] if x not in visited and x not in used_interiors and x not in branches]
                # Exclude other branches from interior
                # Note: u and v are in branches, but they are endpoints.
                # Interior nodes cannot be branches.
                
                # Sort by distance to target? Too expensive.
                # Sort by degree
                neighbors.sort(key=lambda x: degrees[x])
                
                for nei in neighbors:
                    path.append(nei)
                    visited.add(nei)
                    dfs_paths(nei, target, path, visited)
                    path.pop()
                    visited.remove(nei)
                    if len(paths_found) >= max_paths_to_try:
                        return
            
            # Start DFS from u
            # visited should include u and all other branches (except v, which is target)
            # Actually, visited for the path search should prevent revisiting nodes in the path.
            # Forbidden nodes for the path interior: used_interiors + (branches - {u, v})
            
            forbidden_for_interior = set(used_interiors)
            for b in branches:
                if b != u and b != v:
                    forbidden_for_interior.add(b)
            
            # The path can start at u and end at v.
            # Intermediate nodes cannot be in forbidden_for_interior.
            # Also intermediate nodes cannot be u or v (simple path).
            
            # We run DFS from u to v.
            # Visited set for DFS: starts with {u}
            # When expanding, we check if neighbor is in forbidden_for_interior (if it's not v).
            # If neighbor is v, we accept.
            
            def dfs_paths_v2(node, target, path, visited):
                if len(paths_found) >= max_paths_to_try:
                    return
                if node == target:
                    paths_found.append(list(path))
                    return
                
                neighbors = list(adj[node])
                # Shuffle to get different paths in different calls? 
                # No, we want to find *different* paths in this single call to try them.
                # So we just iterate all.
                
                for nei in neighbors:
                    if nei in visited:
                        continue
                    if nei == target:
                        path.append(nei)
                        paths_found.append(list(path))
                        path.pop()
                        if len(paths_found) >= max_paths_to_try:
                            return
                    else:
                        if nei in forbidden_for_interior:
                            continue
                        path.append(nei)
                        visited.add(nei)
                        dfs_paths_v2(nei, target, path, visited)
                        path.pop()
                        visited.remove(nei)
                        if len(paths_found) >= max_paths_to_try:
                            return
            
            path = [u]
            visited = {u}
            dfs_paths_v2(u, v, path, visited)
            
            if not paths_found:
                return None
            
            # Try each found path
            # Sort paths by length (shortest first)
            paths_found.sort(key=len)
            
            for p in paths_found:
                new_used = set(used_interiors)
                new_used.update(p[1:-1])
                
                # Recurse
                result = backtrack(remaining_pairs[1:], new_used, { (u,v): p, **current_paths })
                if result is not None:
                    return result
            
            return None

        # Initial call
        # Shuffle pairs to try different orderings?
        # We can run backtrack multiple times with shuffled pairs if the first attempt fails.
        
        best_result = None
        # Try a few times with different pair orderings
        for _ in range(10):
            if not check_time():
                break
            shuffled_pairs = list(pairs)
            random.shuffle(shuffled_pairs)
            
            # The backtrack function assumes remaining_pairs[0] is the next one.
            # We need to pass the shuffled list.
            # But the function logic above takes `remaining_pairs` and uses `[0]`.
            # So we just call it with the shuffled list.
            
            # Note: The `current_paths` dict keys are (u,v). The order in the list doesn't matter for the dict key.
            res = backtrack(shuffled_pairs, set(), {})
            if res is not None:
                best_result = res
                break
        
        return best_result

    def is_valid_solution(branches, paths_dict):
        if len(branches) < 2:
            return False
        if len(set(branches)) != len(branches):
            return False
        
        used_interiors = set()
        for key, path in paths_dict.items():
            if len(path) < 2:
                return False
            u, v = key
            if path[0] not in (u, v) or path[-1] not in (u, v):
                return False
            if path[0] == path[-1]:
                return False
            if len(set(path)) != len(path):
                return False
            
            # Check adjacency
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
            
            # Check interiors
            interiors = path[1:-1]
            for node in interiors:
                if node in branches:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(interiors)
        
        return True

    # Strategy:
    # 1. Find Max Clique.
    # 2. Try to solve for Max Clique.
    # 3. If fail, try subsets of Max Clique.
    # 4. If fail, try other cliques.
    # 5. If fail, try random sets.

    max_clique = find_max_clique()
    
    best_branches = []
    best_paths = {}
    
    # Try the max clique
    if len(max_clique) >= 2:
        paths = find_disjoint_paths(max_clique)
        if paths and is_valid_solution(max_clique, paths):
            best_branches = max_clique
            best_paths = paths
    
    # If max clique didn't work, try subsets of it
    if not best_branches:
        # Sort clique by degree desc
        clique_sorted = sorted(max_clique, key=lambda x: degrees[x], reverse=True)
        for size in range(len(clique_sorted) - 1, 1, -1):
            if not check_time():
                break
            subset = clique_sorted[:size]
            paths = find_disjoint_paths(subset)
            if paths and is_valid_solution(subset, paths):
                best_branches = subset
                best_paths = paths
                break

    # If still nothing, try to find other large cliques or use greedy
    # Let's also try a greedy expansion from the best found so far, or from scratch if empty.
    
    # If we found something, try to expand it
    if best_branches:
        current_branches = list(best_branches)
        current_paths = best_paths
        
        # Try to add one more vertex
        for _ in range(50):
            if not check_time():
                break
            candidates = [i for i in range(n) if i not in current_branches]
            if not candidates:
                break
            
            # Sort candidates by degree
            candidates.sort(key=lambda x: degrees[x], reverse=True)
            
            added = False
            for new_b in candidates[:10]:
                new_branches = current_branches + [new_b]
                new_paths = find_disjoint_paths(new_branches)
                if new_paths and is_valid_solution(new_branches, new_paths):
                    current_branches = new_branches
                    current_paths = new_paths
                    best_branches = current_branches
                    best_paths = current_paths
                    added = True
                    break
            
            if not added:
                break

    # If we still have nothing, or want to try harder, do random restarts
    # Generate random sets of size k, starting from high k
    if len(best_branches) < 5: # Threshold to trigger more search
        for k in range(min(n, 10), 1, -1):
            if not check_time():
                break
            
            # Try a few random sets of size k
            for _ in range(20):
                if not check_time():
                    break
                random_set = random.sample(range(n), k)
                paths = find_disjoint_paths(random_set)
                if paths and is_valid_solution(random_set, paths):
                    if len(random_set) > len(best_branches):
                        best_branches = random_set
                        best_paths = paths
                        break # Found a valid set of size k, move to next k? 
                    # If we found a valid set of size k, we can stop searching for this k if we want the max, 
                    # but we are iterating k downwards. So if we find one, it's the max for this loop iteration.
                    # But we might find a larger one later? No, k is decreasing.
                    # So if we find one for k, we can break the inner loop.
                    break

    # Final check
    if len(best_branches) < 2:
        # Fallback to an edge
        if P['edges']:
            u, v = P['edges'][0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}

    # Construct output
    final_paths = []
    sorted_branches = sorted(best_branches)
    for u, v in combinations(sorted_branches, 2):
        if (u, v) in best_paths:
            final_paths.append([int(x) for x in best_paths[(u, v)]])
        elif (v, u) in best_paths:
            final_paths.append([int(x) for x in best_paths[(v, u)]])
        else:
            # Should not happen if is_valid_solution passed
            pass

    return {
        'branches': [int(x) for x in sorted_branches],
        'paths': final_paths
    }
