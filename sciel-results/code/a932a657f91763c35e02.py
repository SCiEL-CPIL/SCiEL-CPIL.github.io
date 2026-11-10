import time
import random
from collections import defaultdict
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_validity(colors, edges, n):
    """Full validity check."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    distinct_colors = sorted(color_groups.keys())
    for i in range(len(distinct_colors)):
        for j in range(i+1, len(distinct_colors)):
            c1, c2 = distinct_colors[i], distinct_colors[j]
            sub_vertices = set(color_groups[c1]).union(color_groups[c2])
            if len(sub_vertices) < 3:
                continue
            parent = {v: v for v in sub_vertices}
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
                if u in sub_vertices and v in sub_vertices:
                    if not union(u, v):
                        return False
    return True

def _check_pair_forest(c1, c2, color_groups, edges, n):
    """Check if subgraph induced by colors c1 and c2 is a forest."""
    sub_vertices = set(color_groups.get(c1, [])) | set(color_groups.get(c2, []))
    if len(sub_vertices) < 3:
        return True
    parent = {v: v for v in sub_vertices}
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
        if u in sub_vertices and v in sub_vertices:
            if not union(u, v):
                return False
    return True

def _check_bicolored_after_assign(v, c, colors, adj, color_groups, edges, n):
    """Check if assigning color c to v creates any bicolored cycle."""
    neighbor_colors = set()
    for u in adj[v]:
        if colors[u] >= 0 and colors[u] != c:
            neighbor_colors.add(colors[u])
    
    for nc in neighbor_colors:
        sub_verts = set(color_groups.get(c, [])) | set(color_groups.get(nc, []))
        sub_verts.add(v)
        if len(sub_verts) < 3:
            continue
        parent = {x: x for x in sub_verts}
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
        for u, v2 in edges:
            if u in sub_verts and v2 in sub_verts:
                if not union(u, v2):
                    return False
    return True

def _backtracking_ac(n, adj, edges, k, time_limit):
    """Backtracking with forward checking and incremental bicolored cycle checking."""
    start_time = time.time()
    colors = [-1] * n
    color_groups = defaultdict(list)
    
    # Vertex ordering: highest degree first
    vertices_order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    
    def backtrack(idx):
        if time.time() - start_time > time_limit:
            return None
        if idx == len(vertices_order):
            return colors[:]
        
        v = vertices_order[idx]
        
        # Get forbidden colors from neighbors
        neighbor_colors = set()
        for u in adj[v]:
            if colors[u] >= 0:
                neighbor_colors.add(colors[u])
        
        # Try colors: prefer non-neighbor colors first
        color_order = sorted(range(k), key=lambda c: (c in neighbor_colors, c))
        
        for c in color_order:
            if c in neighbor_colors:
                continue
            
            # Assign
            colors[v] = c
            color_groups[c].append(v)
            
            # Check bicolored cycles
            if _check_bicolored_after_assign(v, c, colors, adj, color_groups, edges, n):
                # Forward check: ensure unassigned neighbors still have options
                fc_ok = True
                for u in adj[v]:
                    if colors[u] == -1:
                        forbidden = set()
                        for w in adj[u]:
                            if colors[w] >= 0:
                                forbidden.add(colors[w])
                        if len(forbidden) >= k:
                            fc_ok = False
                            break
                if fc_ok:
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
            
            # Undo
            colors[v] = -1
            color_groups[c].pop()
        
        return None
    
    return backtrack(0)

def _local_search(k, adj, edges, n, time_limit):
    """Local search with simulated annealing."""
    start_time = time.time()
    best_cost = float('inf')
    best_colors = None
    
    def full_cost(colors):
        cost = 0
        for u, v in edges:
            if colors[u] == colors[v]:
                cost += 100
        color_groups = defaultdict(list)
        for i, c in enumerate(colors):
            color_groups[c].append(i)
        distinct = sorted(color_groups.keys())
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                if not _check_pair_forest(distinct[i], distinct[j], color_groups, edges, n):
                    cost += 50
        return cost
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
        
        colors = [random.randint(0, k-1) for _ in range(n)]
        # Greedy repair of adjacent conflicts
        for _ in range(50):
            fixed = False
            for u, v in edges:
                if colors[u] == colors[v]:
                    for c in range(k):
                        if c != colors[u]:
                            ok = all(colors[w] != c for w in adj[v] if w != u)
                            if ok:
                                colors[v] = c
                                fixed = True
                                break
            if not fixed:
                break
        
        current_cost = full_cost(colors)
        if current_cost == 0:
            if _check_validity(colors, edges, n):
                return colors
        
        temp = 5.0
        for step in range(15000):
            if time.time() - start_time > time_limit:
                break
            
            v = random.randint(0, n-1)
            new_c = random.randint(0, k-1)
            if new_c == colors[v]:
                continue
            
            # Adjacent constraint check
            if any(colors[u] == new_c for u in adj[v]):
                continue
            
            old_c = colors[v]
            colors[v] = new_c
            new_cost = full_cost(colors)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.71828, -delta / max(temp, 0.01)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if current_cost == 0:
                        if _check_validity(colors, edges, n):
                            return colors
            else:
                colors[v] = old_c
            
            temp *= 0.9997
        
        if best_cost == 0 and best_colors is not None:
            if _check_validity(best_colors, edges, n):
                return best_colors
    
    return None

def _generate_3color_patterns(n):
    """Generate comprehensive set of structured 3-coloring patterns."""
    patterns = []
    
    # Linear: (a*row + b*col + c) % 3
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0:
                    continue
                p = [(a*(i//10) + b*(i%10) + c) % 3 for i in range(n)]
                patterns.append(p)
    
    # Quadratic: (a*row + b*col + c*row*col + d) % 3
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    p = [(a*(i//10) + b*(i%10) + c*(i//10)*(i%10) + d) % 3 for i in range(n)]
                    patterns.append(p)
    
    # Different modular bases
    for base in [2, 3, 4, 5, 6]:
        for off in range(3):
            patterns.append([((i//10)*base + (i%10) + off) % 3 for i in range(n)])
            patterns.append([((i//10) + (i%10)*base + off) % 3 for i in range(n)])
    
    # XOR patterns
    for off in range(3):
        patterns.append([((i//10) ^ (i%10) ^ off) % 3 for i in range(n)])
    
    # Row-dependent: each row has a base color, columns alternate
    import itertools
    for rc in itertools.product(range(3), repeat=5):
        for shift in range(3):
            p = []
            for i in range(n):
                row = i // 10
                col = i % 10
                p.append((rc[row] + col + shift) % 3)
            patterns.append(p)
    
    # Column-dependent: each column has a base color, rows alternate
    for cc in itertools.product(range(3), repeat=10):
        for shift in range(3):
            p = []
            for i in range(n):
                row = i // 10
                col = i % 10
                p.append((cc[col] + row + shift) % 3)
            patterns.append(p)
    
    return patterns

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    # Phase 1: Try structured 3-coloring patterns with all permutations
    import itertools
    patterns = _generate_3color_patterns(n)
    for p in patterns:
        if _check_validity(p, edges, n):
            return [int(c) for c in p]
        for perm in itertools.permutations([0, 1, 2]):
            p2 = [perm[c] for c in p]
            if _check_validity(p2, edges, n):
                return [int(c) for c in p2]
    
    # Phase 2: Backtracking with AC for 3 colors
    remaining = deadline - time.time()
    if remaining > 5:
        bt = _backtracking_ac(n, adj, edges, 3, min(25, remaining - 3))
        if bt is not None:
            if _check_validity(bt, edges, n):
                return [int(c) for c in bt]
    
    # Phase 3: Local search for 3 colors
    remaining = deadline - time.time()
    if remaining > 10:
        res = _local_search(3, adj, edges, n, min(25, remaining - 5))
        if res is not None:
            if _check_validity(res, edges, n):
                return [int(c) for c in res]
    
    # Phase 4: Backtracking for 4 colors
    remaining = deadline - time.time()
    if remaining > 5:
        bt = _backtracking_ac(n, adj, edges, 4, min(15, remaining - 3))
        if bt is not None:
            if _check_validity(bt, edges, n):
                return [int(c) for c in bt]
    
    # Phase 5: Local search for 4 colors
    remaining = deadline - time.time()
    if remaining > 10:
        res = _local_search(4, adj, edges, n, min(15, remaining - 5))
        if res is not None:
            if _check_validity(res, edges, n):
                return [int(c) for c in res]
    
    # Fallback
    for k in [5, 6, 7, 8]:
        res = [i % k for i in range(n)]
        if _check_validity(res, edges, n):
            return [int(c) for c in res]
    
    return [i % 8 for i in range(n)]
