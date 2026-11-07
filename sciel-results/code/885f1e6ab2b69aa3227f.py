"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Leave buffer for safety

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Calculate degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Candidate branches: top vertices by degree
    # We need at least k branches. Let's pick top 12 as candidates for k up to 6.
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    candidates = sorted_vertices[:12]
    
    # Pairs to connect
    def get_pairs(branches):
        return list(itertools.combinations(branches, 2))

    # Check if a set of branches forms a valid clique subdivision
    def find_subdivision(branches, time_limit):
        branches = list(branches)
        k = len(branches)
        pairs = get_pairs(branches)
        
        # State: set of available internal vertices
        # Branches are not available as internal vertices
        available = set(range(n)) - set(branches)
        
        # We will recursively try to satisfy pairs
        # To speed up, we track which pairs are satisfied
        # Actually, we just need to find paths for all pairs.
        
        # Heuristic: Sort pairs? Or just pick the first unsatisfied one.
        # A better heuristic: pick the pair with the fewest possible paths in the current graph.
        
        def dfs(pairs_remaining, available_verts, current_paths):
            if time.time() > time_limit:
                return None
            
            if not pairs_remaining:
                return current_paths
            
            # Choose a pair to solve next
            # Heuristic: pick the pair with the minimum number of available simple paths (length <= 5)
            best_pair = None
            best_paths = None
            min_path_count = float('inf')
            
            for u, v in pairs_remaining:
                # Find all simple paths from u to v using only available_verts + {u,v}
                # Path length limit to avoid explosion
                paths = find_short_paths(u, v, available_verts, adj, max_len=6)
                if len(paths) == 0:
                    return None
                if len(paths) < min_path_count:
                    min_path_count = len(paths)
                    best_pair = (u, v)
                    best_paths = paths
                    if min_path_count == 1:
                        break
            
            if best_pair is None:
                return None
                
            u, v = best_pair
            new_pairs = [p for p in pairs_remaining if p != best_pair]
            
            # Try each path
            # Sort paths by length (shorter is better as it consumes fewer vertices)
            best_paths.sort(key=lambda p: len(p))
            
            for path in best_paths:
                internal_verts = set(path[1:-1])
                if not internal_verts.issubset(available_verts):
                    continue
                
                new_available = available_verts - internal_verts
                
                # Recurse
                result = dfs(new_pairs, new_available, current_paths + [path])
                if result is not None:
                    return result
            
            return None

        return dfs(pairs, available, [])

    def find_short_paths(u, v, available_verts, adj, max_len):
        # BFS/DFS to find simple paths up to max_len
        # Since we need disjointness, we restrict to available_verts for internal nodes
        # u and v are branches, so they are allowed as endpoints
        paths = []
        # Use DFS with backtracking
        stack = [(u, [u])]
        visited_in_path = {u}
        
        # To avoid infinite loops and explosion, we rely on max_len and small graph
        # But standard DFS with path tracking is better
        def dfs_path(current, path, visited):
            if len(path) > max_len:
                return
            if current == v:
                if len(path) > 1: # Must have at least one edge
                    paths.append(list(path))
                return
            
            for neighbor in adj[current]:
                # Neighbor must be v or in available_verts
                if neighbor == v:
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        dfs_path(v, path, visited)
                        path.pop()
                        visited.remove(v)
                elif neighbor in available_verts and neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs_path(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs_path(u, [u], {u})
        return paths

    # Try k from 6 down to 3
    for k in range(6, 2, -1):
        if time.time() > deadline:
            break
            
        # Try combinations of candidates
        # To reduce search space, we can fix the first vertex or use a heuristic
        # But with 12 candidates, C(12,6) = 924, C(12,5)=792. It's manageable if dfs is fast.
        found = None
        for combo in itertools.combinations(candidates, k):
            if time.time() > deadline:
                break
            result = find_subdivision(combo, deadline)
            if result is not None:
                found = (list(combo), result)
                break
        
        if found:
            branches, paths = found
            # Convert to required types
            branches = [int(b) for b in branches]
            paths = [[int(p) for p in path] for path in paths]
            return {'branches': branches, 'paths': paths}
            
    # Fallback: Find a triangle
    # The seed logic found a triangle. We can just use that or a simple search.
    # Since we tried 3 in the loop above and failed, it means no K3 subdivision exists in candidates?
    # Wait, the loop goes down to 3. If it fails, we need a valid answer.
    # The graph definitely has a triangle (seed found one).
    # Why would it fail? Maybe candidates didn't include the triangle?
    # Top 12 by degree should include most hubs.
    # Let's do a brute force triangle search on all vertices if the above fails.
    
    # Simple triangle search
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        branches = [u, v, w]
                        paths = [[u, v], [v, w], [w, u]]
                        # Check if these are valid simple paths (they are edges)
                        # Interiors are empty, so disjoint.
                        return {'branches': [int(b) for b in branches], 'paths': paths}
                        
    # Should not reach here
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
