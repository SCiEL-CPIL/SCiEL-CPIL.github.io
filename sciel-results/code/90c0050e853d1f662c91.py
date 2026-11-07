"""Rewritten program: fast bicolored-forest coloring via incremental union-find."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # leave 5s margin before 90s kill
    
    # Try increasing number of colors starting from 3
    for k in range(3, 50):
        if time.time() > deadline - 3:
            break
        
        result = _solve_k(n, adj, k, deadline)
        if result is not None:
            # Verify the result is valid
            if _verify(n, adj, result):
                return result
    
    # Fallback: greedy coloring with many colors
    return _greedy_fallback(n, adj)


def _verify(n, adj, color):
    """Verify that the coloring satisfies all constraints."""
    # Check adjacent vertices have different colors
    for u, v in adj[u]:
        if color[u] == color[v]:
            return False
    
    # Check bicolored forest constraint
    colors_used = set(color)
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            # Check subgraph induced by colors c1 and c2 is a forest
            if not _is_forest_pair(n, adj, color, c1, c2):
                return False
    
    return True


def _is_forest_pair(n, adj, color, c1, c2):
    """Check if subgraph induced by vertices with color c1 or c2 is a forest."""
    # Collect vertices with color c1 or c2
    verts = set()
    for v in range(n):
        if color[v] == c1 or color[v] == c2:
            verts.add(v)
    
    if len(verts) <= 1:
        return True
    
    # Use union-find to check for cycles
    parent = {v: v for v in verts}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True
    
    # Check all edges within the induced subgraph
    for v in verts:
        for u in adj[v]:
            if u in verts and u > v:  # avoid double counting
                if not union(v, u):
                    return False
    
    return True


def _solve_k(n, adj, k, deadline):
    """Try to color graph with k colors (0..k-1) satisfying bicolored forest constraint."""
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    color = [-1] * n
    
    # Union-find structures for each color pair
    # uf[(c1, c2)] where c1 < c2, maps vertex -> parent
    uf_parents = {}
    
    def get_uf(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in uf_parents:
            uf_parents[key] = {}
        return uf_parents[key]
    
    def uf_find(uf, x):
        if x not in uf:
            uf[x] = x
            return x
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x
    
    def uf_union(uf, x, y):
        rx = uf_find(uf, x)
        ry = uf_find(uf, y)
        if rx == ry:
            return False
        uf[rx] = ry
        return True
    
    # For each color, track the union-find for edges within that color
    # (needed because same-color edges appear in all pairs involving that color)
    color_uf = {}
    
    def get_color_uf(c):
        if c not in color_uf:
            color_uf[c] = {}
        return color_uf[c]
    
    def backtrack(idx):
        if time.time() > deadline - 2:
            return None
        
        if idx == n:
            return list(color)
        
        v = order[idx]
        
        # Get forbidden colors (neighbors' colors)
        forbidden = set()
        for u in adj[v]:
            if color[u] != -1:
                forbidden.add(color[u])
        
        # Try colors 0 to k-1, preferring colors not used by neighbors
        # Sort: colors not in forbidden first, then by frequency of use
        color_freq = [0] * k
        for u in range(n):
            if color[u] != -1:
                color_freq[color[u]] += 1
        
        # Try colors not in forbidden, ordered by least frequent first
        candidates = []
        for c in range(k):
            if c not in forbidden:
                candidates.append((color_freq[c], c))
        candidates.sort()
        
        for _, c in candidates:
            # Tentatively assign
            color[v] = c
            
            # Check bicolored forest constraints
            valid = True
            
            # Get all colors currently in use
            active_colors = set()
            for u in range(n):
                if color[u] != -1:
                    active_colors.add(color[u])
            
            # For each active color cu != c, check pair {c, cu}
            for cu in active_colors:
                if cu == c:
                    continue
                
                uf = get_uf(c, cu)
                
                # Check edges from v to neighbors with color cu or c
                for u in adj[v]:
                    if color[u] == -1:
                        continue
                    cu_val = color[u]
                    
                    # Edge (v, u) is in subgraph for pair {c, cu}
                    # if both endpoints have color in {c, cu}
                    if cu_val == c:
                        # Both have color c: edge is in pair {c, cu}
                        if not uf_union(uf, v, u):
                            valid = False
                            break
                    elif cu_val == cu:
                        # v has color c, u has color cu: edge is in pair {c, cu}
                        if not uf_union(uf, v, u):
                            valid = False
                            break
                    # else: u has a different color, edge not in this pair
                
                if not valid:
                    break
            
            if not valid:
                color[v] = -1
                continue
            
            # Also need to check: edges from v to neighbors with same color c
            # These edges are in the color-c subgraph, which appears in all pairs {c, c'}
            # But we already handled those above when cu was any active color.
            # However, we need to make sure the color-c induced subgraph itself is acyclic.
            # This is implicitly checked because if there's a cycle in color-c subgraph,
            # then for any active color c', the pair {c, c'} would have that cycle.
            # But if no other color is active yet, we might miss it.
            # Let's add an explicit check for the color-c subgraph being a forest.
            
            # Actually, let's just verify the full constraint at the end.
            # For now, proceed with backtracking.
            
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            color[v] = -1
        
        return None
    
    return backtrack(0)


def _greedy_fallback(n, adj):
    """Greedy fallback: assign distinct colors to minimize conflicts."""
    color = [-1] * n
    next_color = 0
    
    # Order by degree descending
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    for v in order:
        # Find smallest color not used by neighbors
        used = set()
        for u in adj[v]:
            if color[u] != -1:
                used.add(color[u])
        
        while next_color in used:
            next_color += 1
        
        color[v] = next_color
        next_color += 1
    
    return color


# EVOLVE-BLOCK-END
