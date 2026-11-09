"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest_fast(colors, adj, n):
    """Fast check: for each pair of colors, verify induced subgraph is a forest."""
    color_to_vertices = {}
    for i in range(n):
        c = colors[i]
        if c not in color_to_vertices:
            color_to_vertices[c] = []
        color_to_vertices[c].append(i)
    
    colors_used = sorted(color_to_vertices.keys())
    num_colors = len(colors_used)
    
    for ci in range(num_colors):
        for cj in range(ci + 1, num_colors):
            c1, c2 = colors_used[ci], colors_used[cj]
            verts = color_to_vertices[c1] + color_to_vertices[c2]
            if len(verts) < 2:
                continue
            vert_set = set(verts)
            # Union-find on just these vertices
            parent = {}
            for v in verts:
                parent[v] = v
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            cycle_found = False
            for u in verts:
                for v in adj[u]:
                    if v in vert_set and u < v:
                        ru, rv = find(u), find(v)
                        if ru == rv:
                            cycle_found = True
                            break
                        parent[ru] = rv
                if cycle_found:
                    break
            if cycle_found:
                return False
    return True

def _is_valid_coloring(colors, adj, n):
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    return _check_bicolor_forest_fast(colors, adj, n)

def _count_distinct_colors(colors):
    return len(set(colors))

def _backtrack_color(k, adj, n, deadline):
    """Backtracking to find a valid coloring with exactly k colors (0..k-1)."""
    colors = [-1] * n
    # Order vertices by degree (all 4 here), but use a fixed order for determinism
    order = list(range(n))
    
    def bt(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        v = order[idx]
        # Try colors in order
        for c in range(k):
            # Check proper coloring
            valid = True
            for u in adj[v]:
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            colors[v] = c
            # Quick bicolored check: only check pairs involving c
            # Build partial check
            ok = True
            for c2 in range(k):
                if c2 == c:
                    continue
                # Collect vertices with color c or c2
                verts = []
                for i in range(n):
                    if colors[i] == c or colors[i] == c2:
                        verts.append(i)
                if len(verts) < 2:
                    continue
                vs = set(verts)
                par = {x: x for x in verts}
                def fnd(x):
                    while par[x] != x:
                        par[x] = par[par[x]]
                        x = par[x]
                    return x
                cyc = False
                for u in verts:
                    for w in adj[u]:
                        if w in vs and u < w:
                            ru, rw = fnd(u), fnd(w)
                            if ru == rw:
                                cyc = True
                                break
                            par[ru] = rw
                    if cyc:
                        break
                if cyc:
                    ok = False
                    break
            if ok:
                if bt(idx + 1):
                    return True
            colors[v] = -1
        return False
    
    if bt(0):
        return colors
    return None

def _local_search_reduce(colors, adj, n, deadline):
    """Try to reduce number of distinct colors via local search."""
    current = colors[:]
    distinct = _count_distinct_colors(current)
    max_iter = 200
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        # Try reassigning each vertex to a lower color
        for v in range(n):
            old_c = current[v]
            # Try colors 0 to old_c-1, and also try to consolidate
            target_colors = list(range(old_c))
            # Also try other used colors
            used = set(current)
            for tc in used:
                if tc != old_c and tc not in target_colors:
                    target_colors.append(tc)
            
            for tc in target_colors:
                # Check proper
                proper = True
                for u in adj[v]:
                    if current[u] == tc:
                        proper = False
                        break
                if not proper:
                    continue
                current[v] = tc
                if _check_bicolor_forest_fast(current, adj, n):
                    new_d = _count_distinct_colors(current)
                    if new_d < distinct:
                        distinct = new_d
                        improved = True
                        break
                    else:
                        current[v] = old_c
                else:
                    current[v] = old_c
            if improved:
                break
        if not improved:
            break
    
    return current, distinct

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_score = n + 1
    
    # Try small k values first with backtracking
    for k in range(3, 12):
        if time.time() > deadline:
            break
        result = _backtrack_color(k, adj, n, deadline)
        if result is not None:
            # Local search to potentially reduce further
            final, score = _local_search_reduce(result, adj, n, deadline)
            if score < best_score:
                best_score = score
                best_result = final[:]
                if best_score <= 3:
                    break
    
    # If backtracking didn't find good, try pattern-based + local search
    if best_score > 5:
        # Try linear patterns on the toroidal grid
        # Vertex i is at (i//10, i%10) in a 5x10 grid
        for a in range(1, 8):
            for b in range(1, 8):
                for k in range(3, 12):
                    if time.time() > deadline:
                        break
                    colors = [((r * a + c * b) % k) for r in range(5) for c in range(10)]
                    if _is_valid_coloring(colors, adj, n):
                        final, score = _local_search_reduce(colors, adj, n, deadline)
                        if score < best_score:
                            best_score = score
                            best_result = final[:]
                            if best_score <= 3:
                                break
                if best_score <= 3:
                    break
            if best_score <= 3:
                break
    
    # Fallback: random restarts
    if best_score > 5:
        for seed in range(100):
            if time.time() > deadline:
                break
            random.seed(seed)
            k = random.randint(3, 8)
            colors = [random.randint(0, k - 1) for _ in range(n)]
            if _is_valid_coloring(colors, adj, n):
                final, score = _local_search_reduce(colors, adj, n, deadline)
                if score < best_score:
                    best_score = score
                    best_result = final[:]
                    if best_score <= 3:
                        break
    
    # Ultimate fallback: unique coloring
    if best_result is None:
        best_result = list(range(n))
        best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
