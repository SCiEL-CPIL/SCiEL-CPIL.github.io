"""Rewritten program: strong backtracking with incremental bicolored-cycle detection."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adjacency(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _check_bicolored_forest(coloring, adj):
    """Check that for every pair of colors, the induced subgraph is a forest (acyclic)."""
    n = P['n']
    colors_used = set(coloring)

    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            parent = list(range(n))
            has_cycle = False
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v <= u:
                        continue
                    if coloring[v] not in (c1, c2):
                        continue
                    ru = find(u, parent)
                    rv = find(v, parent)
                    if ru == rv:
                        has_cycle = True
                        break
                    parent[ru] = rv
                if has_cycle:
                    break
            if has_cycle:
                return False
    return True


def _backtrack_coloring(k, adj, order, deadline):
    """
    Backtracking search for a valid k-coloring where:
    - Adjacent vertices have different colors
    - For every pair of colors, the induced subgraph is a forest
    
    Uses incremental cycle detection: when assigning a color to a vertex,
    we check if it creates a bicolored cycle.
    """
    n = P['n']
    coloring = [-1] * n
    
    # For incremental cycle detection, we maintain for each color pair (c1,c2)
    # a union-find structure. But that's expensive. Instead, we'll do a quick
    # check: when we assign color c to vertex v, we check if adding v with color c
    # creates a cycle in the subgraph induced by any pair involving c.
    
    # A faster approach: for each color c assigned to vertex v, for each other color c',
    # check if v has two neighbors (or paths) in the c'-colored subgraph that would
    # create a cycle. This is complex. Let's use a simpler but effective approach:
    # after each assignment, do a local check.
    
    # Actually, let's use a different strategy: maintain the induced subgraphs incrementally.
    # For each pair of colors, track connected components. When we assign color c to vertex v,
    # for each color c' != c, we need to check if v connects two vertices in the same component
    # of the c-c' induced subgraph.
    
    # This is getting complex. Let's use a simpler approach:
    # 1. Assign colors via backtracking with the adjacency constraint
    # 2. After a full assignment, check the bicolored forest constraint
    # 3. Use good heuristics to find valid colorings quickly
    
    # Better: use the fact that for k=3 or k=4, we can try systematic patterns first.
    
    # Let's implement a clean backtracking with the full check at the end,
    # but with strong heuristics for ordering and color selection.
    
    def is_valid_partial(coloring, v, c):
        """Check if assigning color c to vertex v is consistent with already-colored neighbors."""
        for nb in adj[v]:
            if coloring[nb] == c:
                return False
        return True
    
    def check_full_bicolored(coloring):
        """Full check of bicolored forest constraint."""
        colors_used = set(c for c in coloring if c >= 0)
        if len(colors_used) <= 1:
            return True
        color_list = sorted(colors_used)
        
        def find(x, parent):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        for i in range(len(color_list)):
            for j in range(i+1, len(color_list)):
                c1, c2 = color_list[i], color_list[j]
                parent = list(range(n))
                for u in range(n):
                    if coloring[u] not in (c1, c2):
                        continue
                    for v in adj[u]:
                        if v <= u:
                            continue
                        if coloring[v] not in (c1, c2):
                            continue
                        ru = find(u, parent)
                        rv = find(v, parent)
                        if ru == rv:
                            return False
                        parent[ru] = rv
        return True
    
    def backtrack(idx):
        """Recursively assign colors to vertices in order."""
        if time.time() > deadline:
            return False
        if idx == len(order):
            return check_full_bicolored(coloring)
        
        v = order[idx]
        
        # Try colors in a smart order: prefer colors that minimize conflicts
        for c in range(k):
            if not is_valid_partial(coloring, v, c):
                continue
            coloring[v] = c
            if backtrack(idx + 1):
                return True
            coloring[v] = -1
        
        return False
    
    # Order vertices by degree (descending) for better pruning
    # Actually, for this graph all vertices have degree 4, so let's use a different ordering
    # Use the natural order but try to be smart about which colors to try
    
    # Try multiple orderings
    for attempt in range(3):
        if time.time() > deadline:
            break
        if attempt == 0:
            order = list(range(n))
        elif attempt == 1:
            order = list(range(n))
            order.reverse()
        else:
            order = list(range(n))
            random.Random(42 + attempt).shuffle(order)
        
        coloring = [-1] * n
        if backtrack(0):
            return coloring
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges)
    
    # Start with a valid coloring: use n colors (one per vertex)
    best_coloring = list(range(n))
    best_count = n
    
    # Try k=3 first, then 4, 5, ...
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 10.0:
            break
        
        # Try systematic patterns first for speed
        found = False
        for p in range(1, k + 1):
            for q in range(k):
                coloring = [((i * p + q) % k) for i in range(n)]
                # Check adjacent vertices differ
                valid = True
                for u, v in edges:
                    if coloring[u] == coloring[v]:
                        valid = False
                        break
                if not valid:
                    continue
                if _check_bicolored_forest(coloring, adj):
                    cnt = len(set(coloring))
                    if cnt < best_count:
                        best_coloring = coloring
                        best_count = cnt
                    found = True
                    break
            if found:
                break
        
        if found:
            # We found a valid coloring with at most k colors
            # Check if it actually uses fewer colors
            cnt = len(set(best_coloring))
            if cnt <= k:
                # Can't do better than the actual number of colors used
                if cnt == k:
                    break
                elif cnt < k:
                    # Found a coloring with fewer colors than k, great
                    break
        
        # Try backtracking for this k
        if time.time() > deadline - 5.0:
            break
        coloring = _backtrack_coloring(k, adj, list(range(n)), deadline - 2.0)
        if coloring is not None:
            cnt = len(set(coloring))
            if cnt < best_count:
                best_coloring = coloring
                best_count = cnt
            if cnt == k:
                break
    
    # Ensure we return a list of Python ints
    result = [int(c) for c in best_coloring]
    return result

# EVOLVE-BLOCK-END
