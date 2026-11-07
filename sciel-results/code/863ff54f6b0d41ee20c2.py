"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

def _build_adj():
    adj = [[] for _ in range(N)]
    for u, v in EDGES:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _validate(colors):
    """Check adjacency and bicolored forest constraints."""
    # Check adjacency
    for u, v in EDGES:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forest
    used = set(colors)
    for c1 in used:
        for c2 in used:
            if c1 >= c2:
                continue
            # Check if subgraph induced by c1, c2 is a forest
            parent = list(range(N))
            rank = [0] * N
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
            for u, v in EDGES:
                if colors[u] in (c1, c2) and colors[v] in (c1, c2):
                    if not union(u, v):
                        return False
    return True

def _try_pattern(k, pattern_func):
    """Try a coloring pattern and validate."""
    colors = [pattern_func(i) for i in range(N)]
    if _validate(colors):
        return colors
    return None

def _local_search(colors, max_iters=5000):
    """Local search to minimize colors."""
    import random
    current = list(colors)
    best = list(current)
    best_count = len(set(current))
    
    for _ in range(max_iters):
        v = random.randint(0, N - 1)
        old_c = current[v]
        # Try changing to a random color
        new_c = random.randint(0, 49)
        if new_c == old_c:
            continue
        # Check adjacency
        adj = _build_adj()
        valid = True
        for u in adj[v]:
            if current[u] == new_c:
                valid = False
                break
        if not valid:
            continue
        current[v] = new_c
        if _validate(current):
            cnt = len(set(current))
            if cnt < best_count:
                best = list(current)
                best_count = cnt
        else:
            current[v] = old_c
    
    return best, best_count

def construct():
    deadline = time.time() + 85.0
    adj = _build_adj()
    
    # Try various structured patterns for small k
    # Pattern: (row + col) % k, (row + 2*col) % k, etc.
    # Vertices 0..49 in a 5x10 grid: vertex i is at row i//10, col i%10
    
    patterns = []
    for k in range(3, 8):
        # Various linear patterns
        for a in range(k):
            for b in range(k):
                for c in range(k):
                    patterns.append((k, lambda i, a=a, b=b, c=c: (a * (i // 10) + b * (i % 10) + c) % k))
        # Checkerboard-like
        patterns.append((k, lambda i: ((i // 10) + (i % 10)) % k))
        patterns.append((k, lambda i: ((i // 10) * 2 + (i % 10)) % k))
        patterns.append((k, lambda i: ((i // 10) + 2 * (i % 10)) % k))
    
    # Try patterns
    for k, func in patterns:
        if time.time() > deadline - 5:
            break
        colors = _try_pattern(k, func)
        if colors is not None:
            return [int(c) for c in colors]
    
    # Fallback: local search from a random valid coloring
    # Start with a proper coloring (use 4 colors in a pattern)
    base = [(i // 10 + i % 10) % 4 for i in range(N)]
    if _validate(base):
        best, best_count = _local_search(base, max_iters=2000)
        return [int(c) for c in best]
    
    # Final fallback: proper coloring with many colors
    return [int(i) for i in range(N)]

# EVOLVE-BLOCK-END
