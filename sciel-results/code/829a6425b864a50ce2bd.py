"""Bicolored-forest coloring of C(50;{1,7}) with fast incremental SA."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_bicolored_forest(colors, edges, c1, c2, n):
    """Check that subgraph induced by colors c1 and c2 is a forest using union-find."""
    in_set = [False] * n
    for i in range(n):
        if colors[i] == c1 or colors[i] == c2:
            in_set[i] = True
    
    # Quick check: count vertices in set
    cnt = sum(in_set)
    if cnt <= 1:
        return True
    
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
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True
    
    for u, v in edges:
        if in_set[u] and in_set[v]:
            if not union(u, v):
                return False
    return True


def _is_valid(colors, edges, n):
    """Full validity check."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    used = set(colors)
    for i, c1 in enumerate(used):
        for c2 in used:
            if c1 >= c2:
                continue
            if not _check_bicolored_forest(colors, edges, c1, c2, n):
                return False
    return True


def _count_violations_full(colors, edges, n):
    """Count total violations."""
    violations = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            violations += 1
    used = set(colors)
    for i, c1 in enumerate(used):
        for c2 in used:
            if c1 >= c2:
                continue
            if not _check_bicolored_forest(colors, edges, c1, c2, n):
                violations += 1
    return violations


def construct():
    start_time = time.time()
    deadline = start_time + 85
    random.seed(42)

    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list as tuples for speed
    edge_list = [(u, v) for u, v in edges]

    best_colors = None
    best_num_colors = n + 1

    # Try structured patterns first
    patterns = []
    for mod in range(2, 12):
        for mult in range(1, mod):
            patterns.append(lambda i, m=mod, mu=mult: (i * mu) % m)
        patterns.append(lambda i, m=mod: i % m)
    # Additional patterns
    patterns.append(lambda i: (i % 2) * 3 + (i // 2) % 3)
    patterns.append(lambda i: (i % 3) * 2 + (i // 3) % 2)
    patterns.append(lambda i: (i % 5) + 5 * (i // 5 % 2))
    patterns.append(lambda i: (i % 7) + 7 * (i // 7 % 2))
    
    for pf in patterns:
        if time.time() > deadline - 10:
            break
        colors = [pf(i) for i in range(n)]
        if _is_valid(colors, edge_list, n):
            distinct = len(set(colors))
            if distinct < best_num_colors:
                best_num_colors = distinct
                best_colors = colors
    
    if best_colors is not None and best_num_colors <= 5:
        return [int(c) for c in best_colors]

    # Simulated annealing with incremental updates
    for num_colors in range(3, min(best_num_colors, 12)):
        if time.time() > deadline - 5:
            break
        
        max_restarts = 30 if num_colors <= 5 else 15
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            colors = [random.randint(0, num_colors - 1) for _ in range(n)]
            current_cost = _count_violations_full(colors, edge_list, n)
            best_local = list(colors)
            best_local_cost = current_cost
            
            T = 8.0
            cooling = 0.9995
            min_T = 0.01
            steps = 0
            max_steps = 30000
            
            while T > min_T and steps < max_steps:
                if time.time() > deadline - 5:
                    break
                steps += 1
                
                v = random.randint(0, n - 1)
                old_c = colors[v]
                new_c = random.randint(0, num_colors - 1)
                if new_c == old_c:
                    continue
                
                # Check neighbor conflict
                neighbor_colors = set(colors[nb] for nb in adj[v])
                if new_c in neighbor_colors:
                    # Still try, but it will likely increase cost
                    pass
                
                colors[v] = new_c
                new_cost = _count_violations_full(colors, edge_list, n)
                delta = new_cost - current_cost
                
                if delta <= 0 or random.random() < 2.718281828 ** (-delta / T):
                    current_cost = new_cost
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local = list(colors)
                else:
                    colors[v] = old_c
                
                T *= cooling
            
            if best_local_cost == 0:
                distinct = len(set(best_local))
                if distinct < best_num_colors:
                    best_num_colors = distinct
                    best_colors = best_local
                break

    if best_colors is not None:
        return [int(c) for c in best_colors]

    # Fallback: greedy with 10 colors
    colors = [None] * n
    for v in range(n):
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] is not None:
                forbidden.add(colors[nb])
        for c in range(10):
            if c not in forbidden:
                colors[v] = c
                break
        if colors[v] is None:
            colors[v] = 0

    return [int(c) for c in colors]
