import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_valid(coloring):
    """Check if coloring is valid: adjacent differ, and no bicolored cycles."""
    n = P['n']
    edges = P['edges']
    
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    
    colors_used = sorted(set(coloring))
    
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            c1, c2 = colors_used[i], colors_used[j]
            parent = list(range(n))
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if union(u, v):
                        return False
    
    return True

def _try_pattern(period, offset_func, n=50):
    """Try a periodic coloring pattern."""
    coloring = [offset_func(i, period) % period for i in range(n)]
    return coloring

def _search_small_k(max_k, adj, n, deadline):
    """Search for valid coloring with up to max_k colors using structured patterns."""
    # Try simple periodic patterns first
    for k in range(2, max_k + 1):
        if time.time() > deadline:
            break
        
        # Pattern 1: i % k
        coloring = [i % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
        
        # Pattern 2: (i // 5) % k (row-based)
        coloring = [(i // 10) % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
        
        # Pattern 3: (i % 10) // 2 % k (column-pair based)
        coloring = [(i % 10) // 2 % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
        
        # Pattern 4: (i // 10 + i % 10) % k (diagonal)
        coloring = [(i // 10 + i % 10) % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
        
        # Pattern 5: (i // 10 * 2 + i % 10) % k
        coloring = [(i // 10 * 2 + i % 10) % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
        
        # Pattern 6: (i // 10 * 3 + i % 10) % k
        coloring = [(i // 10 * 3 + i % 10) % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
        
        # Pattern 7: (i % 10 * 2 + i // 10) % k
        coloring = [(i % 10 * 2 + i // 10) % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
        
        # Pattern 8: (i % 10 * 3 + i // 10) % k
        coloring = [(i % 10 * 3 + i // 10) % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
        
        # Pattern 9: (i % 10 * 4 + i // 10) % k
        coloring = [(i % 10 * 4 + i // 10) % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
        
        # Pattern 10: (i % 10 * 5 + i // 10) % k
        coloring = [(i % 10 * 5 + i // 10) % k for i in range(n)]
        if _check_valid(coloring):
            return coloring
    
    return None

def _backtrack_search(k, adj, n, deadline):
    """Backtracking search with efficient bicolored cycle checking."""
    coloring = [-1] * n
    
    # Use degeneracy ordering (lowest degree first)
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: deg[x])
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    def check_bicolored_fast(v, c):
        """Check if assigning color c to vertex v creates a bicolored cycle."""
        # For each other color d used in neighbors, check if (v,u) edge creates cycle in (c,d) subgraph
        neighbor_colors = set()
        for u in adj[v]:
            if coloring[u] >= 0 and coloring[u] != c:
                neighbor_colors.add(coloring[u])
        
        for d in neighbor_colors:
            # Check if there's a path from v to any neighbor u with color d,
            # using only vertices colored c or d, without using the direct (v,u) edge
            # If such path exists for any u, then adding (v,u) creates a cycle
            
            # BFS from v in (c,d) induced subgraph
            visited = set()
            stack = [v]
            visited.add(v)
            
            while stack:
                cur = stack.pop()
                for w in adj[cur]:
                    if w in visited:
                        continue
                    cw = coloring[w] if w != v else c
                    if cw == c or cw == d:
                        visited.add(w)
                        stack.append(w)
            
            # Check if any neighbor u with color d is reachable from v
            # (excluding the direct edge, but since we're checking reachability,
            # if u is in visited, there's a path, and adding the edge creates a cycle)
            for u in adj[v]:
                if coloring[u] == d and u in visited:
                    return False  # Cycle would be created
        
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        
        # Try colors in order of least constraining value
        for c in range(k):
            # Check proper coloring constraint
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            coloring[v] = c
            
            if check_bicolored_fast(v, c):
                if backtrack(idx + 1):
                    return True
            
            coloring[v] = -1
        
        return False
    
    if backtrack(0):
        return coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    adj = _build_adj()
    
    # First, try structured patterns for small k
    for k in range(2, 8):
        if time.time() > deadline - 10:
            break
        result = _search_small_k(k, adj, n, deadline)
        if result is not None:
            if _check_valid(result):
                return [int(x) for x in result]
    
    # Then try backtracking search for k=3,4,5
    for k in range(3, 6):
        if time.time() > deadline - 5:
            break
        result = _backtrack_search(k, adj, n, deadline)
        if result is not None:
            if _check_valid(result):
                return [int(x) for x in result]
    
    # Fallback: identity coloring (50 colors)
    return [int(i) for i in range(n)]

if __name__ == '__main__':
    result = construct()
    print(result)
    print(len(set(result)))
