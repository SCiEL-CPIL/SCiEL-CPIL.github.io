import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, edges):
    from collections import defaultdict
    n = len(colors)
    color_map = defaultdict(list)
    for v in range(n):
        color_map[colors[v]].append(v)
    
    colors_used = set(colors)
    for i, c1 in enumerate(colors_used):
        for c2 in colors_used:
            if c1 >= c2:
                continue
            verts = [v for v in range(n) if colors[v] in (c1, c2)]
            if len(verts) < 3:
                continue
            vert_set = set(verts)
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            for u in verts:
                find(u)
            for u, v in edges:
                if u in vert_set and v in vert_set:
                    if not union(u, v):
                        return False
    return True

def _validate(colors, edges):
    n = len(colors)
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    if not _check_bicolored_forest(colors, edges):
        return False
    return True

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    order = list(range(n))
    
    for K in range(3, 11):
        colors = [-1] * n
        current_dsus = {}
        
        def find(parent, x):
            if parent[x] != x:
                parent[x] = find(parent, parent[x])
            return parent[x]
            
        def union(parent, rank, x, y):
            px = find(parent, x)
            py = find(parent, y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            
            # Symmetry breaking:
            # The set of colors used so far.
            # If we pick a color that hasn't been used, it must be the smallest unused one.
            used_colors = set(colors[:idx])
            
            for c in range(K):
                # Check proper coloring
                valid = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Symmetry breaking: if c not in used_colors, it must be the smallest unused
                if c not in used_colors:
                    # Check if there is a smaller unused color
                    min_unused = min(set(range(K)) - used_colors)
                    if c != min_unused:
                        continue

                # Check bicolored forest constraint
                nb_by_color = {}
                for nb in adj[v]:
                    if colors[nb] != -1:
                        d = colors[nb]
                        if d not in nb_by_color:
                            nb_by_color[d] = []
                        nb_by_color[d].append(nb)
                
                can_proceed = True
                modified_keys = []
                for d in nb_by_color:
                    key = (min(c, d), max(c, d))
                    if key not in modified_keys:
                        modified_keys.append(key)
                
                # Save states
                saved_states = {}
                newly_created = []
                for key in modified_keys:
                    if key in current_dsus:
                        p, r = current_dsus[key]
                        saved_states[key] = (p[:], r[:])
                    else:
                        p = [i for i in range(n)]
                        r = [0] * n
                        current_dsus[key] = (p, r)
                        newly_created.append(key)
                        
                # Check
                for d, nbs in nb_by_color.items():
                    key = (min(c, d), max(c, d))
                    p, r = current_dsus[key]
                    roots = set()
                    for nb in nbs:
                        root = find(p, nb)
                        if root in roots:
                            can_proceed = False
                            break
                        roots.add(root)
                    if not can_proceed:
                        break
                
                if not can_proceed:
                    # Rollback
                    for key in modified_keys:
                        if key in saved_states:
                            current_dsus[key] = saved_states[key]
                        if key in newly_created:
                            del current_dsus[key]
                    continue
                
                # Apply
                colors[v] = c
                for d, nbs in nb_by_color.items():
                    key = (min(c, d), max(c, d))
                    p, r = current_dsus[key]
                    for nb in nbs:
                        union(p, r, v, nb)
                
                if backtrack(idx + 1):
                    return True
                
                # Rollback
                for key in modified_keys:
                    if key in saved_states:
                        current_dsus[key] = saved_states[key]
                    if key in newly_created:
                        del current_dsus[key]
                
                colors[v] = -1
            
            return False
        
        if backtrack(0):
            if _validate(colors, edges):
                return [int(c) for c in colors]
            
    return list(range(n))
