"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

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

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try to find a 3-coloring with no bicolored cycles
    # Strategy: backtracking with incremental cycle checking
    # For each color pair, track the induced subgraph edges and check for cycles
    
    # We'll use a smarter approach: 
    # 1. Order vertices by degree (all are 4, so use a heuristic)
    # 2. Use backtracking with constraint propagation
    
    # Let's try a different structural approach:
    # The graph is 4-regular on 50 vertices. Let's see if it's bipartite or not.
    # If not bipartite, we need at least 3 colors.
    
    # Check bipartiteness
    visited = [-1] * n
    bipartite = True
    for start in range(n):
        if visited[start] == -1:
            queue = [start]
            visited[start] = 0
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = 1 - visited[u]
                        queue.append(v)
                    elif visited[v] == visited[u]:
                        bipartite = False
    
    if bipartite:
        # 2 colors suffice for proper coloring, but need to check bicolored cycles
        coloring = visited[:]
        if _is_valid(coloring, edges):
            return [int(x) for x in coloring]
    
    # Try 3-coloring with backtracking
    # Key insight: for 3 colors, there are 3 color pairs. Each pair must form a forest.
    # A forest on m vertices has at most m-1 edges.
    
    # Let's use a constraint-based search
    # Order vertices: try to pick vertices that are highly constrained
    
    # Try multiple random orderings and color orderings
    import random
    rng = random.Random(42)
    
    best = list(range(n))
    best_count = n
    
    # Try systematic 3-coloring search
    # Use the fact that for each color pair, the subgraph must be a forest
    # This means: for colors (0,1), edges between 0-1 vertices must form a forest
    # Similarly for (0,2) and (1,2)
    
    # Backtracking with early cycle detection
    colors = [-1] * n
    # For each color pair, maintain a union-find to detect cycles
    # We'll check at the end for simplicity first, then optimize
    
    # Let's try: fix first few vertices, then backtrack
    # Heuristic: try to make each color class roughly equal size
    
    def try_3coloring(order):
        """Try to 3-color vertices in given order with backtracking."""
        colors_local = [-1] * n
        # For each color pair, track edges and check for cycles
        # pair_edges[ci][cj] = list of edges between color ci and cj
        # We'll use a simpler check: after full assignment, verify
        
        # Incremental: for each pair, maintain count of edges and components
        # A forest with m vertices has exactly m - c edges where c is number of components
        # But we need to check per pair
        
        # Simpler: just do full check at leaf, but prune if proper coloring fails
        def bt(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                # Full coloring, check bicolored cycles
                return not _has_bicolored_cycle(colors_local, edges)
            v = order[idx]
            for c in range(3):
                # Check proper coloring: no neighbor has same color
                ok = True
                for u in adj[v]:
                    if colors_local[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                colors_local[v] = c
                if bt(idx + 1):
                    return True
                colors_local[v] = -1
            return False
        
        return bt(0)
    
    # Try several vertex orderings
    # Ordering 1: natural
    order1 = list(range(n))
    if try_3coloring(order1):
        return [int(x) for x in colors]
    
    # Ordering 2: by some heuristic
    # Try ordering that puts highly connected parts first
    # Actually, let's try a different approach: use the graph structure
    
    # The graph seems to be a circulant-like graph. Let's analyze:
    # Vertex 0 connects to: 1, 7, 43, 49
    # Vertex 1 connects to: 0, 2, 8, 44
    # It looks like edges are (i, i+1), (i, i+7), (i, i-7), (i, i-1) mod something?
    # Let's check: 0-1 (diff 1), 0-7 (diff 7), 0-43 (diff -7 or 43), 0-49 (diff -1 or 49)
    # So it's a circulant C(50, {1, 7})? Let's verify: 49 mod 50 = -1, 43 mod 50 = -7
    # Yes! It's the circulant graph C(50, {1, 7})
    
    # For circulant graphs, we can try periodic colorings
    # color[i] = f(i mod p) for small period p
    
    # Try all periodic 3-colorings with period p dividing 50 or small
    for p in range(1, 51):
        if time.time() > deadline:
            break
        if 50 % p != 0 and p > 25:
            continue
        # Try all mappings from Z_p to {0,1,2}
        # But that's 3^p which is too many for large p
        # For small p, enumerate
        if p <= 10:
            for pattern in range(3 ** p):
                if time.time() > deadline:
                    break
                pat = []
                temp = pattern
                for i in range(p):
                    pat.append(temp % 3)
                    temp //= 3
                # Check if this pattern works
                coloring = [pat[i % p] for i in range(n)]
                if _is_valid(coloring, edges):
                    cnt = _count_colors(coloring)
                    if cnt <= 3:
                        return [int(x) for x in coloring]
    
    # Try non-periodic but structured colorings
    # For C(50, {1,7}), try color[i] = (i * a) mod k for various k
    
    for k in range(3, 15):
        if time.time() > deadline:
            break
        for a in range(1, k):
            if time.time() > deadline:
                break
            for b in range(k):
                coloring = [(i * a + b) % k for i in range(n)]
                if _is_valid(coloring, edges):
                    cnt = _count_colors(coloring)
                    if cnt < best_count:
                        best = coloring[:]
                        best_count = cnt
                    if cnt == 3:
                        return [int(x) for x in coloring]
    
    # Try random 3-colorings with repair
    for attempt in range(10000):
        if time.time() > deadline:
            break
        coloring = [rng.randint(0, 2) for _ in range(n)]
        # Check proper coloring
        proper = True
        for u, v in edges:
            if coloring[u] == coloring[v]:
                proper = False
                break
        if not proper:
            continue
        if not _has_bicolored_cycle(coloring, edges):
            return [int(x) for x in coloring]
    
    # Fallback: return best found or 50-coloring
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
