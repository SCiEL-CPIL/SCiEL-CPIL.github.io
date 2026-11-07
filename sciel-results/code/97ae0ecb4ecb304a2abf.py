import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']

def _is_valid(colors):
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    color_pairs = set()
    for u, v in edges:
        a, b = colors[u], colors[v]
        if a != b:
            color_pairs.add((min(a, b), max(a, b)))
    for a, b in color_pairs:
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == a and cv == b) or (cu == b and cv == a):
                if not union(u, v):
                    return False
    return True

def construct():
    deadline = time.time() + 80
    
    # Build adjacency
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree descending, then by index
    order = sorted(range(N), key=lambda v: (-len(adj[v]), v))
    pos = [0] * N
    for i, v in enumerate(order):
        pos[v] = i
    
    # For backtracking, we assign in order[0], order[1], ...
    # We track for each color pair (a,b) a union-find structure
    
    def solve(n_colors):
        colors = [-1] * N
        # uf: dict (a,b) -> parent list
        uf = {}
        
        def uf_find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def uf_union(parent, x, y):
            rx, ry = uf_find(parent, x), uf_find(parent, y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        
        def backtrack(idx):
            if idx == N:
                return True
            if time.time() > deadline:
                return False
            
            v = order[idx]
            
            for c in range(n_colors):
                # Check properness with already-colored neighbors
                ok = True
                for w in adj[v]:
                    if colors[w] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check bicolored cycle: for each colored neighbor w with color cw != c,
                # the pair (c, cw) must not have v and w already connected
                cycle = False
                for w in adj[v]:
                    cw = colors[w]
                    if cw == -1 or cw == c:
                        continue
                    pair = (min(c, cw), max(c, cw))
                    if pair in uf:
                        parent = uf[pair]
                        if uf_find(parent, v) == uf_find(parent, w):
                            cycle = True
                            break
                if cycle:
                    continue
                
                # Assign
                colors[v] = c
                
                # Save old uf state for pairs affected
                old_uf_entries = {}
                for w in adj[v]:
                    cw = colors[w]
                    if cw == -1 or cw == c:
                        continue
                    pair = (min(c, cw), max(c, cw))
                    if pair not in uf:
                        uf[pair] = list(range(N))
                    old_uf_entries[pair] = uf[pair][:]  # copy
                    uf_union(uf[pair], v, w)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                colors[v] = -1
                for pair, old_parent in old_uf_entries.items():
                    uf[pair] = old_parent
                # Remove pairs that were newly created
                for w in adj[v]:
                    cw = colors[w]
                    if cw == -1 or cw == c:
                        continue
                    pair = (min(c, cw), max(c, cw))
                    if pair not in old_uf_entries:
                        # This pair was newly created, remove it
                        if pair in uf:
                            del uf[pair]
            
            return False
        
        return backtrack(0)
    
    for n_colors in [3, 4]:
        if solve(n_colors):
            # Verify
            if _is_valid(colors):
                return [int(c) for c in colors]
    
    # Fallback: known 4-color pattern
    colors = []
    for v in range(50):
        r = v // 5
        c = v % 5
        col = (r % 2) * 2 + (c % 2)
        colors.append(int(col))
    if _is_valid(colors):
        return colors
    return [int(i) for i in range(50)]
