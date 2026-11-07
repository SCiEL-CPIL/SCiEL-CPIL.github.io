"""Few colors without any bicolored cycle - optimized solver."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_validity(colors, edges):
    """Check if coloring is valid: proper + no bicolored cycle."""
    n = len(colors)
    # Check proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check no bicolored cycle using union-find per color pair
    from collections import defaultdict as dd
    pair_edges = dd(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    for key, elist in pair_edges.items():
        # Union-find
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False  # cycle detected
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _try_k_colors(k, edges, n, deadline, seed):
    """Try to find a valid k-coloring using randomized search."""
    rng = random.Random(seed)
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best = None
    best_count = k + 1
    
    # Start with greedy coloring
    def greedy_color():
        colors = [-1] * n
        for i in range(n):
            used = set()
            for nb in adj[i]:
                if colors[nb] >= 0:
                    used.add(colors[nb])
            avail = [c for c in range(k) if c not in used]
            if avail:
                colors[i] = rng.choice(avail)
            else:
                colors[i] = rng.randint(0, k - 1)
        return colors
    
    # Local search
    def local_search(colors, max_iters=5000):
        colors = list(colors)
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Check validity
            if _check_validity(colors, edges):
                return colors
            # Pick a random vertex and try changing its color
            v = rng.randint(0, n - 1)
            old_c = colors[v]
            for new_c in range(k):
                if new_c == old_c:
                    continue
                # Check if neighbor has new_c
                if any(colors[nb] == new_c for nb in adj[v]):
                    continue
                colors[v] = new_c
                if _check_validity(colors, edges):
                    return colors
                # Keep the change if it doesn't create immediate issues, or revert
                # Actually, let's just keep it and continue
                break
            else:
                colors[v] = old_c
        return colors
    
    # Multiple restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        colors = greedy_color()
        colors = local_search(colors, max_iters=2000)
        if _check_validity(colors, edges):
            return colors
        if attempts >= 200:
            break
    
    return None

def _solve_exact_k(k, edges, n, deadline):
    """More systematic search for k colors."""
    rng = random.Random(42)
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Simulated annealing style
    def anneal():
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        # Fix proper coloring
        for i in range(n):
            used = set(colors[nb] for nb in adj[i])
            if colors[i] in used:
                avail = [c for c in range(k) if c not in used]
                if avail:
                    colors[i] = rng.choice(avail)
        
        temp = 1.0
        cooling = 0.999
        for step in range(5000):
            if time.time() > deadline:
                break
            if _check_validity(colors, edges):
                return colors
            v = rng.randint(0, n - 1)
            new_c = rng.randint(0, k - 1)
            if new_c == colors[v]:
                continue
            if any(colors[nb] == new_c for nb in adj[v]):
                continue
            old_c = colors[v]
            colors[v] = new_c
            # Accept (we always accept valid moves)
            temp *= cooling
        return colors
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        result = anneal()
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Try k=3 first (most likely minimum for this graph)
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        # Try multiple seeds
        for seed in range(20):
            if time.time() > deadline:
                break
            result = _try_k_colors(k, edges, n, deadline, seed)
            if result is not None:
                # Verify
                if _check_validity(result, edges):
                    return [int(c) for c in result]
        # Try annealing
        result = _solve_exact_k(k, edges, n, deadline)
        if result is not None:
            if _check_validity(result, edges):
                return [int(c) for c in result]
    
    # Fallback: use more colors
    # Just do a greedy with many colors
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    colors = [0] * n
    for i in range(n):
        used = set(colors[nb] for nb in adj[i])
        c = 0
        while c in used:
            c += 1
        colors[i] = c
    return [int(c) for c in colors]
