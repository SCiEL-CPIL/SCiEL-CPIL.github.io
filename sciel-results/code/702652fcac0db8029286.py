"""Improved approach: Backtracking Disjoint Path Packing."""
# EVOLVE-BLOCK-START
import itertools
import random
import sys
from collections import deque

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = int(P['n'])
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute all-pairs shortest paths (distances) for heuristic ordering
    dist = [[-1] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    parent[s][v] = u
                    q.append(v)
    
    def get_path(src, dst, blocked):
        """Find a simple path from src to dst avoiding blocked vertices (interiors).
        Returns list of vertices or None. Uses BFS on the fly to respect dynamic blocking."""
        if src == dst:
            return [src]
        if dist[src][dst] == -1:
            return None
        # Quick check: if direct edge and not blocked
        if dst in adj[src]:
             # Check if path [src, dst] is valid (no interiors to block)
             return [src, dst]
        
        visited = {src}
        queue = [[src]]
        # Optimization: Limit BFS depth or nodes explored? 
        # For n=24, BFS is fast.
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for neighbor in adj[node]:
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor in visited:
                    continue
                if neighbor in blocked:
                    continue
                visited.add(neighbor)
                queue.append(path + [neighbor])
        return None

    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_result = None
    best_k = 0
    
    # We will try to find the largest k.
    # Strategy: 
    # 1. Try k from min(n, 15) down to 2.
    # 2. For each k, try a set of candidate combinations.
    #    - First, combinations of top-degree vertices.
    #    - Then, random combinations if top-degree fails.
    
    max_k_to_try = min(n, 15)
    
    for k in range(max_k_to_try, 1, -1):
        found_for_k = False
        
        # Generate candidate sets of size k
        candidates = []
        
        # 1. Top degree combinations
        # Limit pool to top 15 to keep combinations manageable
        pool_size = min(n, 15)
        pool = sorted_vertices[:pool_size]
        
        # If k is close to pool_size, we have few combinations.
        # If k is small, we have many.
        # We will sample if the number of combinations is too large.
        num_combs = 1
        # Calculate C(pool_size, k) roughly to decide on sampling
        # For now, just iterate if pool_size <= 15 and k is reasonable.
        # C(15, 7) = 6435. This is fine.
        
        for comb in itertools.combinations(pool, k):
            candidates.append(list(comb))
            
        # 2. Add some random candidates if we want diversity (optional, might be slow)
        # Let's stick to top-degree first.
        
        # Shuffle candidates to break symmetry? No, top-degree is a strong heuristic.
        
        for branches in candidates:
            branches_set = set(branches)
            pairs = list(itertools.combinations(branches, 2))
            
            # Heuristic: Sort pairs by shortest path length (difficulty)
            # Pairs with short paths are more constrained? 
            # Actually, pairs with *long* minimum distances might be easier to separate?
            # Or pairs with *short* distances are "tighter" and should be placed first?
            # Let's sort by distance ascending (tightest first).
            pairs_sorted = sorted(pairs, key=lambda p: dist[p[0]][p[1]] if dist[p[0]][p[1]] != -1 else 999)
            
            # Backtracking search for disjoint paths
            # State: 
            # - assigned_paths: dict from pair to path
            # - used_interiors: set of vertices used as interiors
            
            # We need to assign a path to each pair in pairs_sorted.
            # Recursive function
            def solve(idx, used_interiors, assigned_paths):
                if idx == len(pairs_sorted):
                    return True
                
                u, v = pairs_sorted[idx]
                
                # Try to find a path from u to v
                # Blocked: all branches except u, v + used_interiors
                blocked = branches_set - {u, v}
                blocked.update(used_interiors)
                
                # Get potential paths? 
                # get_path returns the *shortest* path. 
                # This might be too greedy. 
                # We should try a few different paths if the shortest fails later?
                # For now, let's assume shortest is best. 
                # If we want to be more robust, we can find ALL simple paths up to a certain length.
                
                path = get_path(u, v, blocked)
                
                if path:
                    interiors = path[1:-1]
                    # Since get_path uses BFS, interiors are disjoint from blocked by construction.
                    # We need to ensure they are disjoint from *other* paths' interiors.
                    # 'used_interiors' already contains all previously chosen interiors.
                    
                    # Mark used
                    new_used = used_interiors | set(interiors)
                    new_assigned = dict(assigned_paths)
                    new_assigned[(u, v)] = path
                    new_assigned[(v, u)] = path
                    
                    if solve(idx + 1, new_used, new_assigned):
                        return True
                    else:
                        # Backtrack: 
                        # The issue is that get_path only returns ONE path (the shortest).
                        # If the shortest path blocks a future path, we fail.
                        # We need to try other paths.
                        pass
                return False

            # The simple backtracking above fails because it only tries the shortest path.
            # Let's modify get_path to yield multiple paths or try a few variations.
            # Given time constraints, let's try a different heuristic:
            # Randomize the order of neighbors in BFS to get different "shortest" paths?
            # Or, run the solver multiple times with different random seeds for neighbor ordering.
            
            success = False
            paths_dict = {}
            
            for attempt in range(20): # 20 attempts with randomized BFS tie-breaking
                # Shuffle adjacency lists? No, adj is global.
                # We can simulate randomness by shuffling the `blocked` set? No.
                # Let's modify get_path to accept a random seed and shuffle neighbors.
                
                def get_path_randomized(src, dst, blocked, seed):
                    if src == dst:
                        return [src]
                    visited = {src}
                    queue = [[src]]
                    rng = random.Random(seed)
                    while queue:
                        path = queue.pop(0)
                        node = path[-1]
                        neighbors = list(adj[node])
                        rng.shuffle(neighbors)
                        for neighbor in neighbors:
                            if neighbor == dst:
                                return path + [neighbor]
                            if neighbor in visited:
                                continue
                            if neighbor in blocked:
                                continue
                            visited.add(neighbor)
                            queue.append(path + [neighbor])
                    return None

                current_paths = {}
                used_interiors = set()
                valid = True
                
                for u, v in pairs_sorted:
                    blocked = branches_set - {u, v}
                    blocked.update(used_interiors)
                    
                    path = get_path_randomized(u, v, blocked, attempt)
                    if path is None:
                        valid = False
                        break
                    
                    interiors = path[1:-1]
                    if any(x in used_interiors for x in interiors):
                        valid = False
                        break
                        
                    used_interiors.update(interiors)
                    current_paths[(u, v)] = path
                    current_paths[(v, u)] = path
                
                if valid:
                    success = True
                    paths_dict = current_paths
                    break
            
            if success:
                result_paths = []
                for u, v in pairs:
                    result_paths.append(paths_dict[(u, v)])
                
                best_result = {
                    'branches': [int(b) for b in branches],
                    'paths': result_paths
                }
                best_k = k
                found_for_k = True
                break # Found a solution for this k, which is the max possible for this pool
        
        if found_for_k:
            break # Found max k in this pool, no need to check smaller k
    
    if best_result:
        return best_result

    # Fallback
    if edges:
        u, v = edges[0]
        return {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }
    else:
        return {
            'branches': [0, 1],
            'paths': [[0, 1]]
        }
# EVOLVE-BLOCK-END
