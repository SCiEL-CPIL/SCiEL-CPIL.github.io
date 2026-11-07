"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
from collections import defaultdict

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_paths(branches):
        """Try to find vertex-disjoint paths for all pairs in branches."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort pairs by some heuristic - maybe by number of common neighbors
        def pair_key(p):
            u, v = p
            common = len(adj[u] & adj[v] - branch_set)
            return common
        pairs.sort(key=pair_key, reverse=True)
        
        used_interior = set()
        paths = {}
        
        for u, v in pairs:
            # Find a path from u to v avoiding used_interior and branch_set (except u,v)
            forbidden = used_interior | (branch_set - {u, v})
            # BFS/DFS to find simple path
            path = _find_path(u, v, forbidden, adj)
            if path is None:
                return None
            # Mark interior vertices as used
            for node in path[1:-1]:
                used_interior.add(node)
            paths[(u, v)] = path
        
        return [paths[p] for p in pairs]
    
    def _find_path(start, end, forbidden, adj):
        """Find a simple path from start to end avoiding forbidden vertices."""
        if start in forbidden:
            return None
        visited = {start}
        def dfs(node, path):
            if node == end:
                return list(path)
            for neighbor in adj[node]:
                if neighbor in visited:
                    continue
                if neighbor in forbidden and neighbor != end:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                result = dfs(neighbor, path)
                if result is not None:
                    return result
                path.pop()
                visited.discard(neighbor)
            return None
        return dfs(start, [start])
    
    # Try decreasing k
    for k in range(min(n, 12), 1, -1):
        for combo in combinations(range(n), k):
            branches = list(combo)
            result = find_paths(branches)
            if result is not None:
                return {'branches': branches, 'paths': result}
    
    # Fallback: just return an edge
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
