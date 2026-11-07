"""Rewritten program for large clique subdivisions in a pinned graph."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    n = P['n']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Sort vertices by degree descending for candidate ordering
    vertex_order = sorted(range(n), key=lambda x: -degrees[x])
    
    def find_disjoint_paths(branches, used_interiors):
        """Find internally disjoint paths between all pairs of branches.
        Returns dict mapping (u,v) to path list, or None if impossible."""
        paths = {}
        # Sort pairs by some heuristic: prefer pairs with fewer options first
        pairs = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                pairs.append((branches[i], branches[j]))
        
        # Order pairs: those with fewer available interior vertices first
        def pair_score(pair):
            u, v = pair
            # Count available neighbors not in used_interiors and not branches
            avail_u = len(adj[u] - used_interiors - set(branches))
            avail_v = len(adj[v] - used_interiors - set(branches))
            return min(avail_u, avail_v)
        
        pairs.sort(key=pair_score)
        
        for u, v in pairs:
            # BFS for shortest path avoiding used_interiors and other branches
            # except endpoints
            forbidden = used_interiors | (set(branches) - {u, v})
            queue = deque()
            queue.append((u, [u]))
            visited = {u}
            found = None
            while queue:
                node, path = queue.popleft()
                if node == v:
                    found = path
                    break
                for nb in adj[node]:
                    if nb not in visited and nb not in forbidden:
                        visited.add(nb)
                        queue.append((nb, path + [nb]))
            if found is None:
                return None
            # Check interior vertices are disjoint
            interior = set(found[1:-1])
            if interior & used_interiors:
                return None
            # Check no interior is a branch
            if interior & set(branches):
                return None
            used_interiors = used_interiors | interior
            paths[(u, v)] = found
            paths[(v, u)] = found
        return paths
    
    def try_branch_set(branches):
        """Try to find disjoint paths for given branch set."""
        if len(branches) < 2:
            return None
        return find_disjoint_paths(branches, set())
    
    # Greedy construction: start with top vertices, try to add more
    best_result = None
    best_size = 2
    
    # First, try to find a large set by greedy addition
    # Start with the two highest-degree vertices
    initial = [vertex_order[0], vertex_order[1]]
    
    # Try expanding from various starting pairs
    for start_idx in range(min(10, len(vertex_order))):
        for end_idx in range(start_idx+1, min(10, len(vertex_order))):
            if time.time() > deadline:
                break
            branches = [vertex_order[start_idx], vertex_order[end_idx]]
            # Try to add more vertices
            for candidate in vertex_order:
                if candidate in branches:
                    continue
                if time.time() > deadline:
                    break
                new_branches = branches + [candidate]
                result = try_branch_set(new_branches)
                if result is not None:
                    branches = new_branches
                    if len(branches) > best_size:
                        best_size = len(branches)
                        # Convert to proper format
                        path_dict = {}
                        for (u, v), path in result.items():
                            if u < v:
                                path_dict[(u, v)] = path
                        paths_list = []
                        for i in range(len(branches)):
                            for j in range(i+1, len(branches)):
                                u, v = branches[i], branches[j]
                                paths_list.append(list(path_dict[(u, v)]))
                        best_result = {'branches': list(branches), 'paths': paths_list}
    
    if best_result is not None:
        return best_result
    
    # Fallback: find any valid pair
    for u in range(n):
        for v in range(u+1, n):
            if v in adj[u]:
                return {'branches': [u, v], 'paths': [[u, v]]}
    
    # Last resort
    u, v = P['edges'][0]
    return {'branches': [u, v], 'paths': [[u, v]]}
# EVOLVE-BLOCK-END
