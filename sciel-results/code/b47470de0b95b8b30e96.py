import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    adj_mat = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_mat[u][v] = True
        adj_mat[v][u] = True

    def find_shortest_paths(u, v, blocked, limit=20):
        """Find up to 'limit' shortest simple paths from u to v avoiding blocked."""
        # Use BFS to find shortest path length, then DFS to find paths of that length?
        # Or just DFS with depth limit?
        # For small graphs, DFS finding all simple paths is okay if we limit count.
        paths = []
        stack = [(u, [u])]
        # To ensure we find shorter paths first, we can sort neighbors?
        # Or just collect and sort at end.
        while stack and len(paths) < limit:
            node, path = stack.pop()
            if node == v:
                paths.append(path)
                continue
            for nei in adj[node]:
                if nei not in blocked and nei not in path:
                    stack.append((nei, path + [nei]))
        
        # Sort by length
        paths.sort(key=len)
        return paths

    def check_set(B):
        k = len(B)
        if k < 2:
            return None
        
        # Pre-filter
        for b in B:
            neighbors_out = [x for x in adj[b] if x not in B]
            if len(neighbors_out) < k - 1:
                return None

        pairs = list(combinations(B, 2))
        
        # Sort pairs to process "harder" ones first?
        # Harder = fewer available paths?
        # Let's just use fixed order for now.
        
        used_internal = set()
        found_paths = []
        
        def backtrack(idx):
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            blocked = used_internal | (set(B) - {u, v})
            
            candidate_paths = find_shortest_paths(u, v, blocked)
            
            for path in candidate_paths:
                internal_verts = set(path[1:-1])
                if internal_verts & used_internal:
                    continue
                
                used_internal.update(internal_verts)
                found_paths.append(path)
                
                if backtrack(idx + 1):
                    return True
                
                used_internal.difference_update(internal_verts)
                found_paths.pop()
            
            return False

        if backtrack(0):
            return {pair: path for pair, path in zip(pairs, found_paths)}
        return None

    # Search
    # Try k from 8 down to 3
    # Use a time guard
    global_start = time.time()
    
    for k in range(8, 2, -1):
        # Iterate combinations
        # To speed up, we can skip combinations that fail the pre-filter quickly
        for combo in combinations(range(n), k):
            if time.time() - global_start > 50:
                break
                
            B = list(combo)
            # Quick pre-filter before calling check_set
            valid_candidate = True
            for b in B:
                neighbors_out = [x for x in adj[b] if x not in B]
                if len(neighbors_out) < k - 1:
                    valid_candidate = False
                    break
            if not valid_candidate:
                continue
                
            result = check_set(B)
            if result:
                branches = B
                paths = []
                for (u, v), path in result.items():
                    paths.append(path)
                
                return {
                    'branches': branches,
                    'paths': paths
                }

    # Fallback
    if edges:
        u, v = edges[0]
        return {
            'branches': [u, v],
            'paths': [[u, v]]
        }
    return {
        'branches': [0, 1],
        'paths': [[0, 1]]
    }
