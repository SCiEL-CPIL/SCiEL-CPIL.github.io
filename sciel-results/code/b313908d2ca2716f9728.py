"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _has_bicolored_cycle(coloring, n, edges):
    """Check if any pair of colors induces a cycle using union-find."""
    from collections import defaultdict
    color_pairs = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            color_pairs[pair].append((u, v))
    
    for pair, elist in color_pairs.items():
        # Union-find on vertices involved in this color pair
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
                return False  # cycle detected
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return True
    return False

def _is_valid(coloring, n, edges):
    """Check adjacent vertices differ and no bicolored cycles."""
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return not _has_bicolored_cycle(coloring, n, edges)

def _num_colors(coloring):
    return len(set(coloring))

def _local_search(coloring, n, edges, max_iters=5000):
    """Try to reduce number of colors by reassigning vertices."""
    coloring = list(coloring)
    for _ in range(max_iters):
        improved = False
        for v in range(n):
            cur_c = coloring[v]
            # Try to reassign v to a color that reduces total distinct colors
            # or at least maintains validity
            neighbors_colors = set()
            for u, w in edges:
                if u == v:
                    neighbors_colors.add(coloring[w])
                elif w == v:
                    neighbors_colors.add(coloring[u])
            
            # Try each color from 0 to current max
            max_c = max(coloring)
            for new_c in range(max_c + 1):
                if new_c in neighbors_colors:
                    continue
                # Check if this reduces distinct colors
                # Count how many vertices use cur_c
                count_cur = sum(1 for x in coloring if x == cur_c)
                if new_c == cur_c:
                    continue
                # Would removing cur_c reduce distinct colors?
                if count_cur == 1 and new_c not in set(coloring) - {cur_c}:
                    # This would eliminate cur_c entirely
                    pass
                # Try the assignment
                old_c = coloring[v]
                coloring[v] = new_c
                if _is_valid(coloring, n, edges):
                    nc = _num_colors(coloring)
                    oc = _num_colors([x if x != old_c else x for x in coloring])
                    # Actually just check if we improved
                    if nc < _num_colors([x if x == old_c else x for x in coloring]):
                        improved = True
                        break
                    else:
                        # Keep if same or try next
                        if nc <= _num_colors([x if x == old_c else x for x in coloring]):
                            # Accept to allow exploration
                            pass
                        else:
                            coloring[v] = old_c
                            continue
                else:
                    coloring[v] = old_c
            # End color loop
        if not improved:
            break
    return coloring

def _greedy_color(n, edges, max_colors):
    """Greedy coloring trying to use at most max_colors colors."""
    coloring = [-1] * n
    # Order vertices by degree (descending) for better greedy
    deg = [0] * n
    adj = [[] for _ in range(n)]
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        adj[u].append(v)
        adj[v].append(u)
    
    order = sorted(range(n), key=lambda x: -deg[x])
    
    for v in order:
        used = set()
        for u in adj[v]:
            if coloring[u] >= 0:
                used.add(coloring[u])
        for c in range(max_colors):
            if c not in used:
                coloring[v] = c
                break
        else:
            # Couldn't assign, return None
            return None
    return coloring

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = P['n']
    edges = P['edges']
    
    best_coloring = None
    best_nc = n + 1
    
    # Start with a valid high-color coloring
    base = list(range(n))
    best_coloring = base
    best_nc = n
    
    # Try to find colorings with fewer colors
    random.seed(42)
    
    # Phase 1: Try greedy with increasing color counts
    for max_c in range(2, 10):
        if time.time() > deadline - 10:
            break
        # Try multiple random orderings
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            coloring = _greedy_color(n, edges, max_c)
            if coloring is not None and _is_valid(coloring, n, edges):
                nc = _num_colors(coloring)
                if nc < best_nc:
                    best_nc = nc
                    best_coloring = coloring
                if nc <= max_c:
                    break
    
    # Phase 2: Local search from best coloring
    if time.time() < deadline - 5:
        # Try local search from current best
        cs = _local_search(best_coloring, n, edges, max_iters=2000)
        if _is_valid(cs, n, edges):
            nc = _num_colors(cs)
            if nc < best_nc:
                best_nc = nc
                best_coloring = cs
    
    # Phase 3: More aggressive search - try random restarts with local search
    if time.time() < deadline - 5:
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            # Random valid coloring with limited colors
            max_c = max(3, best_nc - 1) if best_nc > 3 else 3
            coloring = None
            for trial in range(100):
                coloring = _greedy_color(n, edges, max_c)
                if coloring is not None and _is_valid(coloring, n, edges):
                    break
            if coloring is not None:
                cs = _local_search(coloring, n, edges, max_iters=1000)
                if _is_valid(cs, n, edges):
                    nc = _num_colors(cs)
                    if nc < best_nc:
                        best_nc = nc
                        best_coloring = cs
    
    # Final: ensure valid
    if not _is_valid(best_coloring, n, edges):
        best_coloring = list(range(n))
    
    # Ensure Python ints
    return [int(c) for c in best_coloring]

# EVOLVE-BLOCK-END
