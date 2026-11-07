import time
import sys
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def verify(colors, k):
        """Verify coloring is valid: proper + no bicolored cycles."""
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        for i in range(k):
            for j in range(i + 1, k):
                verts = [x for x in range(n) if colors[x] in (i, j)]
                if len(verts) < 3:
                    continue
                parent = {v: v for v in verts}
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
                vs = set(verts)
                for u in verts:
                    for v in adj[u]:
                        if v in vs and u < v:
                            if not union(u, v):
                                return False
        return True

    # Try to find a valid 3-coloring using backtracking with forward checking
    # The key insight: we need a proper 3-coloring where each color-pair subgraph is a forest
    
    def solve_3coloring():
        """Backtracking with DSU-based forward checking for 3-coloring."""
        k = 3
        colors = [-1] * n
        
        # DSU for each color pair: (0,1), (0,2), (1,2)
        # Each DSU is a dict: vertex -> parent
        dsus = [{} for _ in range(3)]  # index 0: (0,1), 1: (0,2), 2: (1,2)
        
        def dsu_find(dsu, x):
            if x not in dsu:
                dsu[x] = x
                return x
            root = x
            while dsu[root] != root:
                root = dsu[root]
            return root
        
        def dsu_union(dsu, x, y):
            rx = dsu_find(dsu, x)
            ry = dsu_find(dsu, y)
            if rx == ry:
                return False
            dsu[rx] = ry
            return True
        
        # Variable ordering: use most-constrained-first
        # Precompute degree
        degrees = [len(adj[i]) for i in range(n)]
        
        # Try different orderings
        orderings = [
            list(range(n)),
            sorted(range(n), key=lambda x: -degrees[x]),
            sorted(range(n), key=lambda x: x % 7),
            list(range(n))[::-1],
        ]
        
        nodes_visited = [0]
        max_nodes = 5000000
        
        for order in orderings:
            if time.time() > deadline - 5:
                break
            
            # Reset
            colors = [-1] * n
            dsus = [{} for _ in range(3)]
            nodes_visited[0] = 0
            
            def bt(pos):
                if nodes_visited[0] > max_nodes:
                    return False
                if time.time() > deadline - 3:
                    return False
                if pos == n:
                    return True
                
                v = order[pos]
                
                # Determine available colors
                available = []
                for c in range(k):
                    # Check proper coloring
                    ok = True
                    for nb in adj[v]:
                        if colors[nb] == c:
                            ok = False
                            break
                    if ok:
                        available.append(c)
                
                # Try each available color
                for c in available:
                    # Check bicolored cycle constraints
                    conflict = False
                    undo_info = []
                    
                    for c2 in range(k):
                        if c2 == c:
                            continue
                        # Find the DSU index for pair (c, c2)
                        pair = tuple(sorted([c, c2]))
                        dsu_idx = pair[0] * 2 + pair[1] if pair[0] < pair[1] else pair[1] * 2 + pair[0]
                        # Actually let's use a cleaner mapping
                        if pair == (0, 1):
                            dsu_idx = 0
                        elif pair == (0, 2):
                            dsu_idx = 1
                        elif pair == (1, 2):
                            dsu_idx = 2
                        
                        dsu = dsus[dsu_idx]
                        
                        # Find neighbors of v with color c2
                        nbs_c2 = [nb for nb in adj[v] if colors[nb] == c2]
                        
                        if len(nbs_c2) < 2:
                            # At most one edge to add, no cycle possible
                            if nbs_c2:
                                nb = nbs_c2[0]
                                rv = dsu_find(dsu, v)
                                rn = dsu_find(dsu, nb)
                                if rv != rn:
                                    undo_info.append((dsu_idx, rv, dsu.get(rv)))
                                    dsu[rv] = rn
                        else:
                            # Multiple neighbors: check if they're all in different components
                            roots = set()
                            for nb in nbs_c2:
                                r = dsu_find(dsu, nb)
                                if r in roots:
                                    conflict = True
                                    break
                                roots.add(r)
                            if conflict:
                                break
                            # Also check if v would create a cycle
                            rv = dsu_find(dsu, v)
                            if rv in roots:
                                conflict = True
                                break
                            # Union v with all nbs_c2
                            for nb in nbs_c2:
                                rn = dsu_find(dsu, nb)
                                if dsu_find(dsu, v) == rn:
                                    conflict = True
                                    break
                                undo_info.append((dsu_idx, dsu_find(dsu, v), dsu.get(dsu_find(dsu, v))))
                                dsu[dsu_find(dsu, v)] = rn
                            if conflict:
                                break
                    
                    if conflict:
                        # Undo any partial unions
                        for dsu_idx, rv, old_val in reversed(undo_info):
                            dsu = dsus[dsu_idx]
                            if old_val is None:
                                del dsu[rv]
                            else:
                                dsu[rv] = old_val
                        continue
                    
                    # Assign color
                    colors[v] = c
                    nodes_visited[0] += 1
                    
                    if bt(pos + 1):
                        return True
                    
                    # Undo
                    for dsu_idx, rv, old_val in reversed(undo_info):
                        dsu = dsus[dsu_idx]
                        if old_val is None:
                            del dsu[rv]
                        else:
                            dsu[rv] = old_val
                    colors[v] = -1
                
                return False
            
            if bt(0):
                if verify(colors, 3):
                    return colors
            # Reset for next ordering
            continue
        
        return None
    
    # First try backtracking for 3-coloring
    result = solve_3coloring()
    if result is not None:
        return [int(c) for c in result]
    
    # Fallback: try structured patterns
    for k in range(3, 8):
        if time.time() > deadline - 5:
            break
        for a in range(1, k):
            for b in range(k):
                colors = [(a * i + b) % k for i in range(n)]
                if verify(colors, k):
                    return [int(c) for c in colors]
    
    # Fallback: local search
    rng = random.Random(42)
    best_colors = list(range(n))
    best_k = n
    
    for k in range(3, 6):
        if time.time() > deadline - 5:
            break
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            colors = [rng.randint(0, k-1) for _ in range(n)]
            for it in range(5000):
                if time.time() > deadline - 3:
                    break
                v = rng.randint(0, n-1)
                old_c = colors[v]
                best_new_c = old_c
                best_viol = None
                
                for nc in range(k):
                    if nc == old_c:
                        continue
                    trial = colors[:]
                    trial[v] = nc
                    # Quick check
                    viol = 0
                    for u, w in edges:
                        if trial[u] == trial[w]:
                            viol += 1
                    if viol == 0:
                        # Check bicolored cycles
                        valid = True
                        for i in range(k):
                            for j in range(i+1, k):
                                verts = [x for x in range(n) if trial[x] in (i, j)]
                                if len(verts) < 3:
                                    continue
                                parent = {v2: v2 for v2 in verts}
                                def find2(x):
                                    while parent[x] != x:
                                        parent[x] = parent[parent[x]]
                                        x = parent[x]
                                    return x
                                def union2(x, y):
                                    rx, ry = find2(x), find2(y)
                                    if rx == ry:
                                        return False
                                    parent[rx] = ry
                                    return True
                                vs = set(verts)
                                for u2 in verts:
                                    for w2 in adj[u2]:
                                        if w2 in vs and u2 < w2:
                                            if not union2(u2, w2):
                                                valid = False
                                                break
                                    if not valid:
                                        break
                            if not valid:
                                break
                        if valid:
                            distinct = len(set(trial))
                            if best_viol is None or distinct < best_viol:
                                best_viol = distinct
                                best_new_c = nc
                
                if best_new_c != old_c:
                    colors[v] = best_new_c
                    if best_viol is not None and best_viol < best_k:
                        best_k = best_viol
                        best_colors = colors[:]
    
    if best_k < n and verify(best_colors, best_k):
        return [int(c) for c in best_colors]
    
    # Final fallback: identity coloring
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
