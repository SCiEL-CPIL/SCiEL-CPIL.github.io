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

# Precompute edge index for fast lookup
edge_index = {}
for idx, (u, v) in enumerate(edges):
    edge_index[(u, v)] = idx
    edge_index[(v, u)] = idx

def is_forest_color_pair(colors, a, b):
    """Check if subgraph induced by colors a and b is a forest."""
    # Count edges between a and b
    parent = {}
    rank = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
            rank[x] = 0
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px = find(x)
        py = find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if (cu == a and cv == b) or (cu == b and cv == a):
            if not union(u, v):
                return False
    return True

def get_cost_fast(colors, K):
    """Fast cost: 100 per same-color edge + 10 per bicolored cycle."""
    cost = 0
    used = set()
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            cost += 100
        else:
            used.add(cu)
            used.add(cv)
    if cost > 0:
        return cost
    cl = list(used)
    for i in range(len(cl)):
        for j in range(i+1, len(cl)):
            if not is_forest_color_pair(colors, cl[i], cl[j]):
                cost += 10
    return cost

def get_cost_incremental(colors, K, changed_vertex, old_color, new_color):
    """Compute cost change when changing one vertex's color."""
    # Check same-color violations
    cost = 0
    # Old color edges that were same-color are now different
    # New color edges that are same-color are new violations
    for v in adj[changed_vertex]:
        if colors[v] == new_color:
            cost += 100
    if cost > 0:
        return cost
    
    # Check bicolored cycles - only need to check pairs involving old or new color
    # This is an approximation for speed; full check done separately
    used = set(colors)
    cl = list(used)
    for i in range(len(cl)):
        for j in range(i+1, len(cl)):
            a, b = cl[i], cl[j]
            if a == old_color or b == old_color or a == new_color or b == new_color:
                if not is_forest_color_pair(colors, a, b):
                    cost += 10
    return cost

def full_cost_check(colors, K):
    """Full cost check for verification."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return 100
    used = set(colors)
    cl = list(used)
    cost = 0
    for i in range(len(cl)):
        for j in range(i+1, len(cl)):
            if not is_forest_color_pair(colors, cl[i], cl[j]):
                cost += 10
    return cost

def local_search(K, time_limit, init_colors=None, deadline=None):
    """Local search with optional initial coloring."""
    start_time = time.time()
    if deadline is None:
        deadline = start_time + time_limit
    best_cost = float('inf')
    best_coloring = None
    
    if init_colors is not None:
        colors = list(init_colors)
        current_cost = full_cost_check(colors, K)
        if current_cost == 0:
            return colors
        no_improve = 0
        max_no_improve = 200
        while time.time() < deadline and no_improve < max_no_improve:
            if current_cost == 0:
                return colors
            v = random.randint(0, N - 1)
            old_c = colors[v]
            best_c = old_c
            min_c = current_cost
            for c in range(K):
                if c == old_c:
                    continue
                colors[v] = c
                c_cost = full_cost_check(colors, K)
                if c_cost < min_c:
                    min_c = c_cost
                    best_c = c
            colors[v] = best_c
            current_cost = min_c
            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = list(colors)
            if current_cost == 0:
                return colors
            if best_c == old_c:
                no_improve += 1
            else:
                no_improve = 0
        if best_cost == 0 and best_coloring:
            return best_coloring
        return None
    
    n_restarts = 0
    max_restarts = 5000
    
    while time.time() < deadline and n_restarts < max_restarts:
        n_restarts += 1
        if n_restarts % 5 == 0:
            a = random.randint(1, K - 1) if K > 1 else 1
            b = random.randint(0, K - 1)
            colors = [(a * i + b) % K for i in range(N)]
        elif n_restarts % 5 == 1:
            colors = [i % K for i in range(N)]
        elif n_restarts % 5 == 2:
            colors = [(i * 2) % K for i in range(N)]
        elif n_restarts % 5 == 3:
            colors = [(i * 3) % K for i in range(N)]
        else:
            colors = [random.randint(0, K - 1) for _ in range(N)]
        
        current_cost = full_cost_check(colors, K)
        if current_cost == 0:
            return colors
            
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline and no_improve < max_no_improve:
            if current_cost == 0:
                return colors
            v = random.randint(0, N - 1)
            old_c = colors[v]
            best_c = old_c
            min_c = current_cost
            for c in range(K):
                if c == old_c:
                    continue
                colors[v] = c
                c_cost = full_cost_check(colors, K)
                if c_cost < min_c:
                    min_c = c_cost
                    best_c = c
            colors[v] = best_c
            current_cost = min_c
            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = list(colors)
            if current_cost == 0:
                return colors
            if best_c == old_c:
                no_improve += 1
            else:
                no_improve = 0
                
    if best_cost == 0 and best_coloring:
        return best_coloring
    return None

def try_structured(K, deadline):
    """Try many structured colorings."""
    # Try all linear modular colorings
    for a in range(1, K):
        for b in range(K):
            if time.time() >= deadline:
                break
            colors = [(a * i + b) % K for i in range(N)]
            if full_cost_check(colors, K) == 0:
                return colors
    # Try quadratic
    for a in range(1, K):
        for b in range(K):
            for c in range(K):
                if time.time() >= deadline:
                    break
                colors = [(a * i * i + b * i + c) % K for i in range(N)]
                if full_cost_check(colors, K) == 0:
                    return colors
    # Try floor-based
    for div in range(1, K + 1):
        for offset in range(K):
            if time.time() >= deadline:
                break
            colors = [((i // div) + offset) % K for i in range(N)]
            if full_cost_check(colors, K) == 0:
                return colors
    return None

def construct():
    deadline = time.time() + 55.0
    
    # First try structured approaches for K=3
    res = try_structured(3, min(deadline, time.time() + 5.0))
    if res is not None:
        return res
    
    # Local search for K=3
    remaining = deadline - time.time()
    if remaining > 5:
        res = local_search(3, min(10.0, remaining - 2.0), deadline=deadline)
        if res is not None:
            return res
    
    # Try K=4
    remaining = deadline - time.time()
    if remaining > 5:
        res = try_structured(4, min(deadline, time.time() + 3.0))
        if res is not None:
            return res
        remaining = deadline - time.time()
        if remaining > 3:
            res = local_search(4, min(15.0, remaining - 2.0), deadline=deadline)
            if res is not None:
                return res
    
    # Try K=5
    remaining = deadline - time.time()
    if remaining > 3:
        res = local_search(5, min(10.0, remaining - 2.0), deadline=deadline)
        if res is not None:
            return res
    
    # Final fallback: try more structured for K=3,4
    for K in [3, 4]:
        for a in range(1, K):
            for b in range(K):
                colors = [(a * i + b) % K for i in range(N)]
                if full_cost_check(colors, K) == 0:
                    return colors
    
    # Ultimate fallback
    return [i % 50 for i in range(N)]

# EVOLVE-BLOCK-END
