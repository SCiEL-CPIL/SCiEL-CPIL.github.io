"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

# Precompute adjacency
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def has_bicolored_cycle(colors):
    """Check if any pair of colors induces a subgraph with a cycle using union-find."""
    k = max(colors) + 1
    for a in range(k):
        for b in range(a+1, k):
            # Union-find on vertices colored a or b
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
                    return False  # cycle
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not union(u, v):
                        return True
                elif cu == a and cv == a:
                    if not union(u, v):
                        return True
                elif cu == b and cv == b:
                    if not union(u, v):
                        return True
    return False

def is_valid(colors):
    """Check all constraints: proper coloring and no bicolored cycles."""
    # Proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # No bicolored cycles
    return not has_bicolored_cycle(colors)

def count_violations(colors):
    """Count proper coloring violations and bicolored cycle violations."""
    violations = 0
    # Proper coloring violations
    for u, v in edges:
        if colors[u] == colors[v]:
            violations += 1
    # Bicolored cycle violations
    k = max(colors) + 1 if colors else 0
    for a in range(k):
        for b in range(a+1, k):
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
                    return 1  # cycle found
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return 0
            
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    violations += union(u, v)
                elif cu == a and cv == a:
                    violations += union(u, v)
                elif cu == b and cv == b:
                    violations += union(u, v)
    return violations

def local_search(k, initial=None, max_iter=3000, seed=None):
    """Local search to find valid coloring with k colors."""
    if seed is not None:
        random.seed(seed)
    
    if initial is None:
        colors = [random.randint(0, k-1) for _ in range(n)]
    else:
        colors = list(initial)
    
    best_cost = float('inf')
    best_colors = list(colors)
    
    current_cost = count_violations(colors)
    
    for it in range(max_iter):
        if current_cost == 0:
            return colors
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = list(colors)
        
        # Pick random vertex to change
        v = random.randint(0, n-1)
        old = colors[v]
        best_new = old
        best_c = current_cost
        
        for new_c in range(k):
            if new_c == old:
                continue
            colors[v] = new_c
            nc = count_violations(colors)
            if nc < best_c:
                best_c = nc
                best_new = new_c
        colors[v] = best_new
        current_cost = best_c
    
    if best_cost == 0:
        return best_colors
    return None

def construct():
    deadline = time.time() + 85  # 85s out of 90s budget
    
    # Try k = 3, 4, 5, ...
    best_result = None
    
    for k in range(3, 12):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts
        found = None
        max_restarts = 200
        for attempt in range(max_restarts):
            if time.time() > deadline - 3:
                break
            result = local_search(k, max_iter=1500, seed=random.randint(0, 2**31))
            if result is not None:
                found = result
                break
        
        if found is not None:
            best_result = found
            break
    
    if best_result is None:
        # Fallback: use all distinct colors (always valid)
        return list(range(n))
    
    # Ensure Python ints
    return [int(c) for c in best_result]

# EVOLVE-BLOCK-END
