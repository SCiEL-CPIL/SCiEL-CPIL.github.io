"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

# Precompute adjacency
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def find_bicolored_cycle(colors, k):
    """Find one bicolored cycle. Returns (color_a, color_b, cycle_vertices) or None."""
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    
    used_colors = [c for c in range(k) if c in color_groups]
    
    for a, b in combinations(used_colors, 2):
        relevant_edges = []
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == a or cu == b) and (cv == a or cv == b):
                relevant_edges.append((u, v))
        
        if len(relevant_edges) < 3:
            continue
        
        parent = {}
        for u, v in relevant_edges:
            if u not in parent:
                parent[u] = u
            if v not in parent:
                parent[v] = v
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        for u, v in relevant_edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                # Found a cycle - reconstruct it
                # Walk back from u and v to find the cycle path
                path_u = [u]
                cur = u
                while cur != ru:
                    cur = parent[cur]
                    path_u.append(cur)
                path_v = [v]
                cur = v
                while cur != ru:
                    cur = parent[cur]
                    path_v.append(cur)
                path_v.reverse()
                cycle = path_u + path_v
                return (a, b, cycle)
            parent[ru] = rv
    
    return None

def count_violations_fast(colors, k):
    """Fast count: proper coloring conflicts + number of bicolored cycles."""
    violations = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            violations += 10
    
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    
    used_colors = [c for c in range(k) if c in color_groups]
    
    for a, b in combinations(used_colors, 2):
        relevant_edges = []
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == a or cu == b) and (cv == a or cv == b):
                relevant_edges.append((u, v))
        
        if len(relevant_edges) < 3:
            continue
        
        parent = {}
        rank = {}
        for u, v in relevant_edges:
            if u not in parent:
                parent[u] = u
                rank[u] = 0
            if v not in parent:
                parent[v] = v
                rank[v] = 0
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        for u, v in relevant_edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                violations += 1
            else:
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
    
    return violations

def cycle_directed_search(k, initial=None, max_iter=3000, seed=None, deadline=None):
    """
    Constraint-satisfaction local search that specifically targets bicolored cycles.
    When a cycle is found, recolor a vertex on the cycle to a third color.
    """
    if seed is not None:
        random.seed(seed)
    
    if initial is None:
        colors = [random.randint(0, k-1) for _ in range(n)]
    else:
        colors = list(initial)
    
    current_cost = count_violations_fast(colors, k)
    best_cost = current_cost
    best_colors = list(colors)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if current_cost == 0:
            return colors
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = list(colors)
        
        # Find a bicolored cycle to target
        cycle_info = find_bicolored_cycle(colors, k)
        
        if cycle_info is not None:
            ca, cb, cycle_verts = cycle_info
            # Pick a random vertex on the cycle to recolor
            v = random.choice(cycle_verts)
            old = colors[v]
            
            # Try recoloring to a third color (not ca or cb)
            third_colors = [c for c in range(k) if c != ca and c != cb]
            if not third_colors:
                third_colors = [c for c in range(k) if c != old]
            
            best_new = old
            best_c = current_cost
            
            for new_c in third_colors:
                # Check if new_c would create a proper coloring conflict
                conflict = False
                for nb in adj[v]:
                    if colors[nb] == new_c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                colors[v] = new_c
                nc = count_violations_fast(colors, k)
                if nc < best_c:
                    best_c = nc
                    best_new = new_c
            colors[v] = best_new
            current_cost = best_c
        else:
            # No bicolored cycle found, but still have conflicts (proper coloring)
            # Do a generic move: pick a random vertex with a conflict
            conflict_verts = []
            for u, v in edges:
                if colors[u] == colors[v]:
                    if u not in conflict_verts:
                        conflict_verts.append(u)
            
            if conflict_verts:
                v = random.choice(conflict_verts)
            else:
                v = random.randint(0, n-1)
            
            old = colors[v]
            best_new = old
            best_c = current_cost
            
            for new_c in range(k):
                if new_c == old:
                    continue
                colors[v] = new_c
                nc = count_violations_fast(colors, k)
                if nc < best_c:
                    best_c = nc
                    best_new = new_c
            colors[v] = best_new
            current_cost = best_c
    
    if best_cost == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 45
    
    best_result = None
    best_k = 50
    
    for k in range(3, 15):
        if time.time() > deadline - 5:
            break
        
        found = None
        
        # Multiple restarts with cycle-directed search
        max_restarts = 30
        for attempt in range(max_restarts):
            if time.time() > deadline - 2:
                break
            seed = random.randint(0, 2**31)
            result = cycle_directed_search(k, max_iter=3000, seed=seed, deadline=deadline - 1)
            if result is not None:
                found = result
                break
        
        if found is not None:
            if count_violations_fast(found, k) == 0:
                best_result = found
                best_k = k
                break
    
    if best_result is None:
        # Fallback: try with higher k
        for k_try in range(10, 20):
            if time.time() > deadline - 1:
                break
            for attempt in range(10):
                if time.time() > deadline - 0.5:
                    break
                seed = random.randint(0, 2**31)
                result = cycle_directed_search(k_try, max_iter=2000, seed=seed, deadline=deadline - 0.3)
                if result is not None and count_violations_fast(result, k_try) == 0:
                    best_result = result
                    best_k = k_try
                    break
            if best_result is not None:
                break
    
    if best_result is None:
        return list(range(n))
    
    return [int(c) for c in best_result]

# EVOLVE-BLOCK-END
