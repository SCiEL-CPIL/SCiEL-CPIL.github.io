import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _verify(coloring):
    n = P['n']
    edges = P['edges']
    # Check adjacency
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    
    # Check bicolored cycles
    colors_used = set(coloring)
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            # Union-Find for subgraph induced by c1, c2
            parent = list(range(n))
            rank = [0] * n
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for u, v in edges:
                if (coloring[u] == c1 or coloring[u] == c2) and (coloring[v] == c1 or coloring[v] == c2):
                    if not union(u, v):
                        return False
    return True

def _solve_k3():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (all are 4, so arbitrary or by index)
    # Using index order might be fine, or we can use a heuristic
    order = list(range(n))
    
    color = [-1] * n
    start_time = time.time()
    deadline = start_time + 5.0 # 5 seconds for k=3
    
    # Precompute edges for fast access
    edges = P['edges']
    
    # For efficient cycle checking, we can maintain union-find structures for each pair
    # But since k=3, there are only 3 pairs: (0,1), (0,2), (1,2)
    # We can re-check only the affected pairs when a vertex is colored.
    
    def check_pair(c1, c2):
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        for u, v in edges:
            if (color[u] == c1 or color[u] == c2) and (color[v] == c1 or color[v] == c2):
                if not union(u, v):
                    return False
        return True

    def dfs(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        used = set()
        for nb in adj[v]:
            if color[nb] != -1:
                used.add(color[nb])
        
        for c in range(3):
            if c in used:
                continue
            
            color[v] = c
            
            # Check if this assignment creates a bicolored cycle
            valid = True
            # Only need to check pairs involving c
            for other_c in [0, 1, 2]:
                if other_c == c:
                    continue
                # If other_c is not used yet, no cycle can be formed involving it
                # But we check anyway for simplicity, or optimize by checking if other_c is present
                # For now, just check
                if not check_pair(c, other_c):
                    valid = False
                    break
            
            if valid:
                if dfs(idx + 1):
                    return True
            
            color[v] = -1
        
        return False

    if dfs(0):
        return color[:]
    return None

def construct():
    # Try to find a 3-coloring
    res = _solve_k3()
    if res is not None:
        if _verify(res):
            return [int(c) for c in res]
    
    # Fallback to 4-coloring: c(i) = i % 4
    n = P['n']
    color = [i % 4 for i in range(n)]
    if _verify(color):
        return [int(c) for c in color]
    
    # Fallback to 5-coloring if 4 fails (unlikely)
    color = [i % 5 for i in range(n)]
    if _verify(color):
        return [int(c) for c in color]
        
    # Final fallback: distinct colors
    return [int(i) for i in range(n)]
