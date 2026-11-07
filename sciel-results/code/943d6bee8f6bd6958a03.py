"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import sys
from collections import deque

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Pre-compute degrees for heuristic
    degrees = [len(adj[i]) for i in range(n)]

    def find_paths(u, v, blocked, max_paths=10):
        """
        Find up to 'max_paths' simple paths from u to v avoiding 'blocked' (except u, v).
        Returns a list of paths, sorted by 'cost' (sum of degrees of internal nodes).
        Lower cost is preferred (uses low-degree nodes).
        """
        if u == v:
            return [[u]]
        
        paths = []
        # Use DFS to find simple paths
        # To avoid exponential explosion, limit depth or use BFS layers
        # Given N=24, DFS with pruning is okay if we stop early.
        
        def dfs(curr, path, visited):
            if len(paths) >= max_paths:
                return
            if curr == v:
                paths.append(list(path))
                return
            for nxt in adj[curr]:
                if nxt in visited:
                    continue
                if nxt in blocked and nxt not in (u, v):
                    continue
                visited.add(nxt)
                path.append(nxt)
                dfs(nxt, path, visited)
                path.pop()
                visited.remove(nxt)
                if len(paths) >= max_paths:
                    return

        visited = {u}
        dfs(u, [u], visited)
        
        # Sort paths by cost: sum of degrees of internal nodes
        def path_cost(p):
            if len(p) <= 2:
                return 0
            return sum(degrees[x] for x in p[1:-1])
        
        paths.sort(key=path_cost)
        return paths

    def check_branches(branches):
        """Check if there are disjoint paths for all pairs in branches."""
        pairs = list(itertools.combinations(branches, 2))
        
        # Heuristic: Sort pairs by "difficulty"
        # Difficulty: Number of available internal nodes between them?
        # Or just keep original order? 
        # Let's try sorting by the number of common neighbors (fewer common neighbors = harder?)
        # Actually, pairs that are "far" might be easier?
        # Let's stick to a simple heuristic: Pairs involving the highest degree vertices first?
        # No, let's just use the order generated.
        
        def solve(idx, used, current_paths):
            if idx == len(pairs):
                return True, current_paths
            
            u, v = pairs[idx]
            blocked = used | (set(branches) - {u, v})
            
            candidates = find_paths(u, v, blocked)
            
            for path in candidates:
                internal = set(path[1:-1])
                if not (internal & used):
                    new_used = used | internal
                    res, paths = solve(idx + 1, new_used, current_paths + [path])
                    if res:
                        return True, paths
            
            return False, None

        res, paths = solve(0, set(), [])
        if res:
            return paths
        return None

    best_branches = None
    best_paths = None
    
    vertices = list(range(n))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    # Try to find the largest k
    # Start from a high estimate. Max possible is limited by N.
    # With N=24, k=8 is a good target.
    for k in range(8, 2, -1):
        if k > n:
            continue
        # Limit combinations to avoid timeout
        # If k is large, combinations are many.
        limit = 5000
        count = 0
        for combo in itertools.combinations(vertices, k):
            if count >= limit:
                break
            count += 1
            res = check_branches(combo)
            if res:
                best_branches = list(combo)
                best_paths = res
                break
        if best_branches:
            break
            
    if not best_branches:
        # Fallback
        u, v = P['edges'][0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    return {'branches': best_branches, 'paths': best_paths}
# EVOLVE-BLOCK-END
