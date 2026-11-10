"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

# Set recursion limit to handle deep backtracking if needed, though n=24 is small
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Sort vertices by degree descending to prioritize high-connectivity nodes
    vertices = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    best_k = 0
    best_result = None
    start_time = time.time()
    time_limit = 50.0 # Safety margin
    
    def find_disjoint_paths(src, targets, forbidden):
        """
        Try to find pairwise internally disjoint paths from src to each t in targets.
        The paths must not use any vertex in 'forbidden' (interior vertices).
        Endpoints (src and targets) are allowed to be in forbidden? 
        No, 'forbidden' should contain all previously used interior vertices AND all branch vertices except src and the specific target being processed?
        Actually, the constraint is:
        1. Path interiors are disjoint from each other.
        2. Path interiors are disjoint from all branches (including src and targets? No, endpoints are branches).
        3. Path interiors are disjoint from previously found path interiors.
        
        So, for the new paths from src to targets:
        - The interior vertices of each path must not be in 'forbidden' (which contains all existing branch vertices except src and the target? No, all existing branches are forbidden as interiors).
        - The interior vertices of the new paths must be disjoint from each other.
        
        Let's implement a backtracking search for the paths.
        We pick targets one by one.
        """
        # forbidden_set contains vertices that cannot be used as INTERIOR vertices.
        # This includes all existing branches (except src? src is endpoint, so it's ok. Targets are endpoints, so they are ok).
        # But wait, if a target is in forbidden_set, that's fine because it's an endpoint.
        # However, we must ensure that the path doesn't pass through another branch.
        # So, forbidden_interiors = forbidden_set - {src} - set(targets)
        
        # Actually, simpler: 
        # 'blocked' set contains vertices that cannot be visited as interior nodes.
        # This includes:
        # 1. All vertices in 'forbidden' (previous interiors + all branches except src and targets? No, all branches are blocked as interiors).
        # Let's define 'blocked' = forbidden | set(all_branches_so_far) - {src} - set(targets)
        # But 'forbidden' passed in should be the union of previous interiors.
        # And we know current_branches.
        
        pass # Placeholder, we will implement the logic in the main search
        
    # We will use a different approach for the check:
    # When adding a new vertex 'v' to 'branches', we need to find paths from 'v' to all 'u' in 'branches'.
    # Let's call the set of existing branches 'B_old'.
    # We need paths P_u from v to u for each u in B_old.
    # Constraints:
    # 1. interiors(P_u) are pairwise disjoint.
    # 2. interiors(P_u) are disjoint from 'used_interior' (global set of used interiors).
    # 3. interiors(P_u) do not contain any vertex in B_old ∪ {v} except the endpoints.
    
    def find_k_shortest_paths(src, dst, blocked, k=5):
        """
        Find up to k shortest simple paths from src to dst avoiding blocked vertices.
        Returns a list of paths.
        """
        from collections import deque
        paths = []
        # We will use a priority queue (Dijkstra-like) but for simple paths?
        # Dijkstra doesn't handle simple paths directly if we don't track visited.
        # Since N is small, we can just use BFS to find the shortest, then block its interior, find next, etc.
        # This is a heuristic for "k shortest" but works well for disjoint path finding.
        
        current_blocked = set(blocked)
        
        for _ in range(k):
            q = deque()
            q.append((src, [src]))
            visited = {src}
            found_path = None
            
            while q:
                node, path = q.popleft()
                if node == dst:
                    found_path = path
                    break
                for neighbor in adj[node]:
                    if neighbor in visited:
                        continue
                    if neighbor in current_blocked:
                        continue
                    visited.add(neighbor)
                    q.append((neighbor, path + [neighbor]))
            
            if not found_path:
                break
            
            paths.append(found_path)
            # Block the interior of this path to find the next distinct path
            for v in found_path[1:-1]:
                current_blocked.add(v)
            # Also block the path itself? No, just interiors is enough to force a different route.
            # Actually, to ensure distinct paths, we should block the whole path?
            # If we only block interiors, the next path might be the same length but different vertices.
            # If we block the path vertices, we ensure vertex-disjointness from previous candidates.
            # But we want *different* paths, not necessarily disjoint from each other yet (the DFS will handle that).
            # Blocking interiors is a good heuristic to get diverse paths.
            
        return paths

    def try_add_vertex(v, B_old, used_interior):
        """
        Returns (success, new_paths, new_used_interior)
        new_paths is a dict mapping u to path list.
        """
        if not B_old:
            return True, {}, set()
            
        targets = list(B_old)
        
        def dfs_paths(idx, chosen_paths, local_blocked):
            if idx == len(targets):
                return True, chosen_paths
            
            t = targets[idx]
            forbidden_interiors = used_interior | local_blocked | set(B_old) | {v}
            forbidden_interiors.discard(t)
            
            # Get a few candidate paths
            candidates = find_k_shortest_paths(v, t, forbidden_interiors, k=3)
            
            for path in candidates:
                new_interiors = set(path[1:-1])
                # Ensure new_interiors are actually disjoint from local_blocked (they should be by construction)
                ok, res = dfs_paths(idx + 1, chosen_paths + [(t, path)], local_blocked | new_interiors)
                if ok:
                    return True, res
            
            return False, None
        
        ok, paths_list = dfs_paths(0, [], set())
        if ok:
            paths_dict = {}
            new_interiors = set()
            for t, path in paths_list:
                paths_dict[t] = path
                new_interiors.update(path[1:-1])
            return True, paths_dict, new_interiors
        else:
            return False, None, None

    # Backtracking search for the best set of branches
    # We iterate through vertices in order of degree.
    # We maintain current_branches (list), used_interior (set), and current_paths (dict mapping pair to path? or just list of paths?)
    # Actually, we need to return the paths at the end.
    # Let's store the paths as a list of tuples (u, v, path) or just reconstruct at the end.
    # Reconstructing is easier: if we have the set of branches, we can re-run the path finding?
    # No, the path finding depends on the order of addition.
    # It's better to store the paths found during the addition.
    # Structure: current_paths is a list of (u, v, path) where u,v are branches and path is the list.
    # When adding v to B, we find paths from v to all u in B. We add these to current_paths.
    
    def search(idx, current_branches, used_interior, current_paths):
        nonlocal best_k, best_result, start_time
        
        if time.time() - start_time > time_limit:
            return
            
        k = len(current_branches)
        if k > best_k:
            best_k = k
            # Deep copy the result
            # Branches: list of ints
            # Paths: list of lists
            # The contract says: 'paths': one simple path for each unordered pair of branches.
            # So we need a list of paths.
            res_paths = [path for (_, _, path) in current_paths]
            best_result = {
                'branches': list(current_branches),
                'paths': res_paths
            }
            
        if idx == len(vertices):
            return
        
        v = vertices[idx]
        
        # Option 1: Skip v
        search(idx + 1, current_branches, used_interior, current_paths)
        
        # Option 2: Add v
        # Check if v can be added
        if v in current_branches:
            return # Should not happen
        
        success, new_paths_dict, new_interiors = try_add_vertex(v, current_branches, used_interior)
        
        if success:
            # Update state
            new_branches = current_branches + [v]
            new_used = used_interior | new_interiors
            
            # Update paths list
            # new_paths_dict maps target u to path from v to u.
            # We need to add these to current_paths.
            added_paths = []
            for u, path in new_paths_dict.items():
                # Ensure consistent ordering? The contract says "unordered pair".
                # The path list just needs to contain the paths.
                added_paths.append((v, u, path))
                
            new_paths = current_paths + added_paths
            
            search(idx + 1, new_branches, new_used, new_paths)

    # Start search
    search(0, [], set(), [])
    
    if best_result is None:
        # Fallback: return a valid small solution (e.g., an edge)
        u, v = P['edges'][0]
        best_result = {
            'branches': [u, v],
            'paths': [[u, v]]
        }
        
    # Verification step to ensure validity
    branches = best_result['branches']
    paths = best_result['paths']
    
    # Check 1: At least 2 branches
    if len(branches) < 2:
        u, v = P['edges'][0]
        return {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }

    # Check 2: Paths correspond to all unordered pairs
    expected_pairs = set()
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            expected_pairs.add(frozenset((branches[i], branches[j])))
    
    actual_pairs = set()
    for path in paths:
        if len(path) < 2:
            return {
                'branches': [int(P['edges'][0][0]), int(P['edges'][0][1])],
                'paths': [[int(P['edges'][0][0]), int(P['edges'][0][1])]]
            }
        actual_pairs.add(frozenset((path[0], path[-1])))
        
    if expected_pairs != actual_pairs:
        # Fallback if structure is broken
        u, v = P['edges'][0]
        return {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }

    # Check 3: Disjointness and validity
    branch_set = set(branches)
    used_interiors = set()
    for path in paths:
        # Check adjacency
        for i in range(len(path)-1):
            if path[i+1] not in adj[path[i]]:
                u, v = P['edges'][0]
                return {
                    'branches': [int(u), int(v)],
                    'paths': [[int(u), int(v)]]
                }
        
        interiors = set(path[1:-1])
        # Check interiors don't touch branches
        if interiors & branch_set:
            u, v = P['edges'][0]
            return {
                'branches': [int(u), int(v)],
                'paths': [[int(u), int(v)]]
            }
        
        # Check interiors are disjoint from previous
        if interiors & used_interiors:
            u, v = P['edges'][0]
            return {
                'branches': [int(u), int(v)],
                'paths': [[int(u), int(v)]]
            }
        
        used_interiors.update(interiors)

    # Ensure types are correct
    final_branches = [int(b) for b in best_result['branches']]
    final_paths = [[int(x) for x in path] for path in best_result['paths']]
    
    return {
        'branches': final_branches,
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
