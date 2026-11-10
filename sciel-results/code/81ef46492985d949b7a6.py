import time
import random
from collections import defaultdict
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_validity(colors, edges, n):
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    color_groups = defaultdict(set)
    for i, c in enumerate(colors):
        color_groups[c].add(i)
        
    distinct_colors = list(color_groups.keys())
    
    for i in range(len(distinct_colors)):
        for j in range(i+1, len(distinct_colors)):
            c1 = distinct_colors[i]
            c2 = distinct_colors[j]
            sub_vertices = color_groups[c1] | color_groups[c2]
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

def _calc_cost_fast(colors, edges, n, adj):
    """Fast cost calculation: adjacent conflicts + bicolored cycles"""
    cost = 0
    
    # Adjacent conflicts
    for u, v in edges:
        if colors[u] == colors[v]:
            cost += 10
    
    # Build color groups
    color_groups = defaultdict(set)
    for i, c in enumerate(colors):
        color_groups[c].add(i)
    
    distinct_colors = list(color_groups.keys())
    
    # For each pair, check for cycles using union-find
    for i in range(len(distinct_colors)):
        for j in range(i+1, len(distinct_colors)):
            c1 = distinct_colors[i]
            c2 = distinct_colors[j]
            sub_vertices = color_groups[c1] | color_groups[c2]
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
                        cost += 5
                        break  # One cycle is enough to penalize
    
    return cost

def _local_search(k, adj, edges, n, time_limit):
    start_time = time.time()
    best_cost = float('inf')
    best_colors = None
    
    deadline = start_time + time_limit
    
    def try_coloring(colors):
        nonlocal best_cost, best_colors
        cost = _calc_cost_fast(colors, edges, n, adj)
        if cost == 0:
            return colors
        if cost < best_cost:
            best_cost = cost
            best_colors = colors[:]
        return None
    
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Structured initialization: use a base pattern then perturb
        if r % 3 == 0:
            colors = [i % k for i in range(n)]
        elif r % 3 == 1:
            colors = [(i * 7) % k for i in range(n)]
        else:
            colors = [random.randint(0, k-1) for _ in range(n)]
        
        result = try_coloring(colors)
        if result is not None:
            return result
        
        # Simulated annealing
        temp = 5.0
        current_cost = _calc_cost_fast(colors, edges, n, adj)
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Pick vertex with most conflicts, or random
            if step % 5 == 0:
                # Find vertex with most conflicts
                conflicts = [0] * n
                for u, v in edges:
                    if colors[u] == colors[v]:
                        conflicts[u] += 1
                        conflicts[v] += 1
                v = max(range(n), key=lambda x: conflicts[x])
                if conflicts[v] == 0:
                    v = random.randint(0, n-1)
            else:
                v = random.randint(0, n-1)
            
            old_c = colors[v]
            new_c = random.randint(0, k-1)
            if new_c == old_c:
                continue
            
            colors[v] = new_c
            new_cost = _calc_cost_fast(colors, edges, n, adj)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.71828, -delta / max(temp, 0.1)):
                current_cost = new_cost
                if current_cost == 0:
                    return colors
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                colors[v] = old_c
            
            temp *= 0.998
        
        if best_cost == 0:
            return best_colors
    
    if best_cost == 0:
        return best_colors
    return None

def _backtracking_3color(n, adj, edges):
    """Try to find a valid 3-coloring using backtracking with MRV heuristic."""
    colors = [-1] * n
    
    # Precompute degree
    deg = [len(adj[v]) for v in range(n)]
    
    def get_mrv_vertex():
        """Select uncolored vertex with fewest available colors (MRV)."""
        best_v = -1
        best_count = float('inf')
        for v in range(n):
            if colors[v] != -1:
                continue
            # Count available colors
            used = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            avail = 3 - len(used)
            if avail < best_count:
                best_count = avail
                best_v = v
                if best_count <= 0:
                    break
        return best_v, best_count
    
    def has_bicolored_cycle_partial(colors):
        """Check if current partial coloring creates any bicolored cycle."""
        color_groups = defaultdict(set)
        for i, c in enumerate(colors):
            if c >= 0:
                color_groups[c].add(i)
        
        distinct_colors = list(color_groups.keys())
        
        for i in range(len(distinct_colors)):
            for j in range(i+1, len(distinct_colors)):
                c1 = distinct_colors[i]
                c2 = distinct_colors[j]
                sub_vertices = color_groups[c1] | color_groups[c2]
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
                            return True
                            
        return False
    
    def backtrack():
        v, avail = get_mrv_vertex()
        if v == -1:
            return True  # All colored
        
        if avail <= 0:
            return False
        
        for c in range(3):
            # Check adjacency constraint
            valid = True
            for u in adj[v]:
                if colors[u] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            colors[v] = c
            
            # Check for bicolored cycles
            if not has_bicolored_cycle_partial(colors):
                if backtrack():
                    return True
            
            colors[v] = -1
        
        return False
    
    if backtrack():
        return colors
    return None

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try 3-coloring with backtracking (time-limited)
    bt_start = time.time()
    bt_result = _backtracking_3color(n, adj, edges)
    if bt_result is not None:
        if _check_validity(bt_result, edges, n):
            return [int(c) for c in bt_result]
    
    # Try local search for k=3, 4, 5 with remaining time
    elapsed = time.time() - bt_start
    remaining = max(1, 75 - elapsed)  # Leave 15s margin
    
    for k in [3, 4, 5]:
        t_left = remaining / (6 - k)  # Distribute time
        res = _local_search(k, adj, edges, n, t_left)
        if res is not None:
            if _check_validity(res, edges, n):
                return [int(c) for c in res]
    
    # Fallback: 6 colors
    res_6 = [i % 6 for i in range(n)]
    if _check_validity(res_6, edges, n):
        return [int(c) for c in res_6]
        
    # Last resort: 7 colors
    return [i % 7 for i in range(n)]
