"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

def _has_bicolored_cycle(coloring, edges):
    """Check if any pair of colors induces a cycle using union-find."""
    n = len(coloring)
    color_map = defaultdict(list)
    for i, c in enumerate(coloring):
        color_map[c].append(i)
    
    colors = list(color_map.keys())
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            ci, cj = colors[i], colors[j]
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
                    return True
                parent[rx] = ry
                return False
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == ci and cv == cj) or (cu == cj and cv == ci):
                    if union(u, v):
                        return True
    return False

def _is_valid(coloring, edges):
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return not _has_bicolored_cycle(coloring, edges)

def _count_colors(coloring):
    return len(set(coloring))

def _solve_k_coloring_fast(k, deadline):
    """
    Fast backtracking with MRV heuristic and incremental DSU.
    Uses vertex ordering by degree and available colors.
    """
    sys.setrecursionlimit(10000)
    
    color = [-1] * N
    pair_dsu = {}
    
    def make_key(a, b):
        if a > b:
            a, b = b, a
        return (a, b)
    
    def find(pk, v):
        d = pair_dsu[pk]
        parent = d
        if v not in parent:
            parent[v] = v
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    
    def union(pk, u, v):
        d = pair_dsu[pk]
        ru = find(pk, u)
        rv = find(pk, v)
        if ru == rv:
            return False
        d[rv] = ru
        return True
    
    def assign(v, c):
        color[v] = c
        saved = []
        for u in ADJ[v]:
            d = color[u]
            if d == -1 or d == c:
                continue
            pk = make_key(c, d)
            if pk not in pair_dsu:
                pair_dsu[pk] = {}
            state_before = dict(pair_dsu[pk])
            if not union(pk, v, u):
                color[v] = -1
                for pk2, st in saved:
                    pair_dsu[pk2] = st
                return (False, [])
            saved.append((pk, state_before))
        return (True, saved)
    
    def unassign(v, saved):
        color[v] = -1
        for pk, st in saved:
            pair_dsu[pk] = st
    
    def get_available_colors(v, k):
        """Get colors not used by neighbors."""
        used = set()
        for u in ADJ[v]:
            if color[u] != -1:
                used.add(color[u])
        return [c for c in range(k) if c not in used]
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        # Find uncolored vertex with fewest available colors (MRV)
        best_v = -1
        best_count = k + 1
        best_colors = None
        
        for v in range(N):
            if color[v] != -1:
                continue
            avail = get_available_colors(v, k)
            if len(avail) < best_count:
                best_count = len(avail)
                best_v = v
                best_colors = avail
                if best_count == 0:
                    return False
                if best_count == 1:
                    break
        
        if best_v == -1:
            return True  # All colored
        
        for c in best_colors:
            success, saved = assign(best_v, c)
            if success:
                if backtrack():
                    return True
                unassign(best_v, saved)
        
        return False
    
    if backtrack():
        return color[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try small k first with fast solver
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        result = _solve_k_coloring_fast(k, deadline)
        if result is not None:
            if _is_valid(result, EDGES):
                return [int(x) for x in result]
    
    # Fallback: try modular patterns
    for k in range(3, 20):
        if time.time() > deadline:
            break
        for a in range(1, k):
            for b in range(k):
                coloring = [(i * a + b) % k for i in range(N)]
                if _is_valid(coloring, EDGES):
                    return [int(x) for x in coloring]
    
    # Last resort: 50 colors
    return [int(i) for i in range(N)]
# EVOLVE-BLOCK-END
