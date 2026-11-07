"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _has_bicolored_cycle(coloring, edges):
    """Check if any pair of colors induces a cycle using union-find."""
    n = len(coloring)
    color_map = {}
    for i, c in enumerate(coloring):
        if c not in color_map:
            color_map[c] = []
        color_map[c].append(i)
    
    colors = list(color_map.keys())
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            ci, cj = colors[i], colors[j]
            # Union-find for vertices with color ci or cj
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True  # cycle detected
                parent[rx] = ry
                return False
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == ci and cv == cj) or (cu == cj and cv == ci):
                    if union(u, v):
                        return True  # bicolored cycle found
    return False

def _is_valid(coloring, edges):
    """Check proper coloring and no bicolored cycles."""
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return not _has_bicolored_cycle(coloring, edges)

def _count_colors(coloring):
    return len(set(coloring))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit, 5s safety margin
    n = P['n']
    edges = P['edges']
    
    # Try k=3 first (we know k=2 fails because whole graph has cycles)
    # Use a systematic search for 3-coloring
    # The graph is a circulant C(50, {1,7})
    # Try periodic colorings: color[i] = f(i mod p) for small p
    
    best = list(range(n))  # fallback: 50 colors, always valid
    best_count = n
    
    # Try 3-colorings based on modular patterns
    # Since graph is circulant, try color[i] = (i * a + b) mod k for various a, b, k
    
    for k in range(3, 10):
        if time.time() > deadline:
            break
        for a in range(1, k):
            if time.time() > deadline:
                break
            for b in range(k):
                if time.time() > deadline:
                    break
                coloring = [(i * a + b) % k for i in range(n)]
                if _is_valid(coloring, edges):
                    cnt = _count_colors(coloring)
                    if cnt < best_count:
                        best = coloring
                        best_count = cnt
                    if best_count == 3:
                        return [int(x) for x in best]
    
    # If modular didn't work, try backtracking for k=3
    if best_count > 3 and time.time() < deadline:
        # Backtracking search for 3-coloring
        colors = [0] * n
        def backtrack(v):
            if time.time() > deadline:
                return False
            if v == n:
                if _has_bicolored_cycle(colors, edges):
                    return False
                return True
            for c in range(3):
                # Check proper coloring
                valid = True
                for u, w in edges:
                    if u == v:
                        if colors[w] == c:
                            valid = False
                            break
                    elif w == v:
                        if colors[u] == c:
                            valid = False
                            break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = 0
            return False
        
        if backtrack(0):
            best = colors[:]
            best_count = 3
    
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
