import time
import random
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def count_bicolored_cycles(colors):
    """Count total number of bicolored cycles (excess edges in each bicolored subgraph)."""
    color_set = set(colors)
    total_excess = 0
    proper_violations = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            proper_violations += 1
    for a in color_set:
        for b in color_set:
            if a >= b:
                continue
            edge_count = 0
            vertex_count = 0
            for u in range(N):
                if colors[u] in (a, b):
                    vertex_count += 1
            for u, v in edges:
                if (colors[u] == a and colors[v] == b) or (colors[u] == b and colors[v] == a):
                    edge_count += 1
            if vertex_count > 0:
                max_forest_edges = vertex_count - 1
            else:
                max_forest_edges = 0
            excess = edge_count - max_forest_edges
            if excess > 0:
                total_excess += excess
    return proper_violations * 100 + total_excess

def is_valid(colors):
    return count_bicolored_cycles(colors) == 0

def check_bicolored_forest(colors, a, b):
    """Check if subgraph induced by colors a and b is a forest using union-find."""
    verts = [u for u in range(N) if colors[u] in (a, b)]
    if len(verts) <= 1:
        return True
    vert_set = set(verts)
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
    for u, v in edges:
        if colors[u] in (a, b) and colors[v] in (a, b):
            if not union(u, v):
                return False
    return True

def construct():
    deadline = time.time() + 85.0
    
    # Phase 1: Try all modular patterns color(i) = (a*i + b) % k for small k
    for k in range(2, 10):
        if time.time() > deadline - 30.0:
            break
        for a in range(1, k):
            for b in range(k):
                if time.time() > deadline - 25.0:
                    break
                colors = [(a * i + b) % k for i in range(N)]
                proper = True
                for u, v in edges:
                    if colors[u] == colors[v]:
                        proper = False
                        break
                if not proper:
                    continue
                valid = True
                for ci in range(k):
                    for cj in range(ci + 1, k):
                        if not check_bicolored_forest(colors, ci, cj):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return [int(c) for c in colors]
    
    # Phase 2: Try linear patterns with larger multipliers
    for k in range(3, 12):
        if time.time() > deadline - 15.0:
            break
        for a in range(1, 50, k):
            for b in range(k):
                if time.time() > deadline - 10.0:
                    break
                colors = [(a * i + b) % k for i in range(N)]
                proper = True
                for u, v in edges:
                    if colors[u] == colors[v]:
                        proper = False
                        break
                if not proper:
                    continue
                valid = True
                for ci in range(k):
                    for cj in range(ci + 1, k):
                        if not check_bicolored_forest(colors, ci, cj):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return [int(c) for c in colors]
    
    # Phase 3: Constraint-satisfaction with backtracking
    # Assign colors greedily, tracking bicolored subgraph connectivity
    # Key: for each color pair (a,b), track union-find to detect cycles
    
    best_score = float('inf')
    best_colors = None
    
    # Try different numbers of colors
    for max_colors in range(3, 8):
        if time.time() > deadline - 5.0:
            break
        
        # Multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 3.0:
                break
            
            random.seed(42 + restart)
            # Shuffle vertex order for diversity
            order = list(range(N))
            random.shuffle(order)
            
            colors = [-1] * N
            # Union-find structures for each color pair
            # pair_uvf[(a,b)] = {vertex: parent} for a < b
            pair_uvf = {}
            pair_edge_count = {}
            
            def get_pair(a, b):
                if a > b:
                    a, b = b, a
                return (a, b)
            
            def pair_find(a, b, x):
                key = get_pair(a, b)
                uf = pair_uvf.get(key)
                if uf is None:
                    return x
                while uf[x] != x:
                    uf[x] = uf[uf[x]]
                    x = uf[x]
                return x
            
            def pair_union(a, b, x, y):
                key = get_pair(a, b)
                if key not in pair_uvf:
                    pair_uvf[key] = {}
                uf = pair_uvf[key]
                if x not in uf:
                    uf[x] = x
                if y not in uf:
                    uf[y] = y
                rx = pair_find(a, b, x)
                ry = pair_find(a, b, y)
                if rx == ry:
                    return False
                uf[rx] = ry
                return True
            
            def can_assign(v, c):
                # Check proper: no neighbor has color c
                for nb in adj[v]:
                    if colors[nb] == c:
                        return False
                # Check bicolored cycle: for each neighbor nb with color d != c
                # the pair (c, d) must not form a cycle
                for nb in adj[v]:
                    d = colors[nb]
                    if d == -1 or d == c:
                        continue
                    if not pair_union(c, d, v, nb):
                        # This would create a cycle in the bicolored subgraph
                        return False
                return True
            
            def assign(v, c):
                colors[v] = c
                for nb in adj[v]:
                    d = colors[nb]
                    if d == -1 or d == c:
                        continue
                    pair_union(c, d, v, nb)
            
            def unassign(v, c):
                # Roll back: we need to undo the unions
                # This is complex; instead we use a different approach
                colors[v] = -1
                # Rebuild pair_uvf from scratch for correctness
                pair_uvf.clear()
                pair_edge_count.clear()
                for i in range(N):
                    if colors[i] == -1:
                        continue
                    for nb in adj[i]:
                        if colors[nb] == -1:
                            continue
                        if colors[i] < colors[nb]:
                            pair_union(colors[i], colors[nb], i, nb)
            
            # Backtracking search
            def backtrack(idx):
                if time.time() > deadline - 2.0:
                    return False
                if idx == N:
                    return True
                
                v = order[idx]
                tried = set()
                for c in range(max_colors):
                    if c in tried:
                        continue
                    if can_assign(v, c):
                        assign(v, c)
                        if backtrack(idx + 1):
                            return True
                        unassign(v, c)
                    tried.add(c)
                return False
            
            if backtrack(0):
                # Verify
                if is_valid(colors):
                    distinct = len(set(colors))
                    return [int(c) for c in colors]
    
    # Phase 4: Simulated annealing with better scoring
    # Start from a valid 3-coloring and try to reduce colors
    base_colors = [i % 3 for i in range(N)]
    if not is_valid(base_colors):
        # Find a valid starting point
        for k in range(3, 8):
            for a in range(1, k):
                for b in range(k):
                    colors = [(a * i + b) % k for i in range(N)]
                    if is_valid(colors):
                        base_colors = colors
                        break
                if is_valid(base_colors):
                    break
            if is_valid(base_colors):
                break
    
    current = base_colors[:]
    current_score = count_bicolored_cycles(current)
    best_score = current_score
    best_colors = current[:]
    temperature = 3.0
    cooling = 0.999
    
    while time.time() < deadline - 2.0:
        v = random.randint(0, N - 1)
        old_color = current[v]
        new_color = random.randint(0, 7)
        if new_color == old_color:
            continue
        
        conflict = False
        for nb in adj[v]:
            if current[nb] == new_color:
                conflict = True
                break
        if conflict:
            continue
        
        current[v] = new_color
        new_score = count_bicolored_cycles(current)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < (1.0 / (1.0 + delta / temperature)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_colors = current[:]
                if best_score == 0:
                    return [int(c) for c in best_colors]
        else:
            current[v] = old_color
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    if best_score == 0:
        return [int(c) for c in best_colors]
    
    # Phase 5: Systematic local search
    current = best_colors[:]
    current_score = best_score
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for v in range(N):
            old_color = current[v]
            for new_color in range(8):
                if new_color == old_color:
                    continue
                conflict = False
                for nb in adj[v]:
                    if current[nb] == new_color:
                        conflict = True
                        break
                if conflict:
                    continue
                current[v] = new_color
                new_score = count_bicolored_cycles(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_colors = current[:]
                    if best_score == 0:
                        return [int(c) for c in best_colors]
                    break
            current[v] = old_color
    
    if best_score == 0:
        return [int(c) for c in best_colors]
    
    if best_colors is not None:
        return [int(c) for c in best_colors]
    
    colors = [0] * N
    for i in range(N):
        used = set()
        for v in adj[i]:
            if v < i:
                used.add(colors[v])
        for c in range(50):
            if c not in used:
                colors[i] = c
                break
    return [int(c) for c in colors]
