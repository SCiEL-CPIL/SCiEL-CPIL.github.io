"""Rewritten program for few-colors-no-bicolored-cycle task."""
import time
import itertools

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_valid(colors, edges):
    """Check if coloring is valid: adjacent differ, and no bicolored cycle."""
    n = len(colors)
    # Check adjacent vertices differ
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check no bicolored cycle using union-find
    # For each pair of colors, check if edges between them form a forest
    color_pairs = {}
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False
        key = (min(cu, cv), max(cu, cv))
        if key not in color_pairs:
            color_pairs[key] = []
        color_pairs[key].append((u, v))
    
    for key, edge_list in color_pairs.items():
        # Union-find to detect cycles in this bicolored subgraph
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
                return False  # cycle detected
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        for u, v in edge_list:
            if not union(u, v):
                return False  # bicolored cycle
    return True

def _try_k_colors(k, edges, n, deadline):
    """Try to find a valid k-coloring using backtracking."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    colors = [-1] * n
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        # Try each color
        for c in range(k):
            # Check if color c is valid for vertex idx
            valid = True
            for nb in adj[idx]:
                if colors[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            colors[idx] = c
            # Check bicolored cycle constraint incrementally is complex;
            # instead, check full validity at the end or use a smarter approach
            # For now, just do full check at leaf
            if backtrack(idx + 1):
                return True
            colors[idx] = -1
        return False
    
    # Use vertex ordering by degree (descending) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = {v: i for i, v in enumerate(order)}
    colors2 = [-1] * n
    
    def backtrack2(pos_idx):
        if time.time() > deadline:
            return False
        if pos_idx == n:
            return True
        v = order[pos_idx]
        for c in range(k):
            valid = True
            for nb in adj[v]:
                if colors2[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            colors2[v] = c
            if backtrack2(pos_idx + 1):
                return True
            colors2[v] = -1
        return False
    
    if backtrack2(0):
        return colors2
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Try k from 2 upward
    for k in range(2, 51):
        result = _try_k_colors(k, edges, n, deadline)
        if result is not None:
            # Verify
            if _check_valid(result, edges):
                return [int(x) for x in result]
    
    # Fallback: return seed coloring
    return [int(x) for x in range(n)]
# EVOLVE-BLOCK-END
