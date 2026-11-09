"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 80  # safety margin
    
    # Try 3 colors first, then 4, etc.
    for max_colors in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 10:
            break
        result = _search(n, adj, max_colors, deadline)
        if result is not None:
            return result
    return list(range(n))

def _search(n, adj, max_colors, deadline):
    """Backtracking search with MRV heuristic."""
    color = [-1] * n
    # For each pair of colors, maintain union-find to detect bicolored cycles
    # pair_uf[(c1,c2)] = {parent, rank}
    pair_ufs = {}
    
    def find(uf, x):
        while uf['parent'][x] != x:
            uf['parent'][x] = uf['parent'][uf['parent'][x]]
            x = uf['parent'][x]
        return x
    
    def union(uf, x, y):
        rx = find(uf, x)
        ry = find(uf, y)
        if rx == ry:
            return False  # cycle detected
        if uf['rank'][rx] < uf['rank'][ry]:
            rx, ry = ry, rx
        uf['parent'][ry] = rx
        if uf['rank'][rx] == uf['rank'][ry]:
            uf['rank'][rx] += 1
        return True
    
    def get_uf(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        key = (c1, c2)
        if key not in pair_ufs:
            pair_ufs[key] = {'parent': list(range(n)), 'rank': [0] * n}
        return pair_ufs[key]
    
    # Undo stack for union operations
    undo_stack = []
    
    def do_union(c1, c2, u, v):
        """Union u,v in the UF for color pair (c1,c2). Returns True if no cycle."""
        if c1 > c2:
            c1, c2 = c2, c1
        key = (c1, c2)
        if key not in pair_ufs:
            pair_ufs[key] = {'parent': list(range(n)), 'rank': [0] * n}
        uf = pair_ufs[key]
        rx = find(uf, u)
        ry = find(uf, v)
        if rx == ry:
            return False
        if uf['rank'][rx] < uf['rank'][ry]:
            rx, ry = ry, rx
        undo_stack.append((key, ry, rx, uf['rank'][rx]))
        uf['parent'][ry] = rx
        if uf['rank'][rx] == uf['rank'][ry]:
            uf['rank'][rx] += 1
        return True
    
    def undo():
        if undo_stack:
            key, ry, rx, old_rank = undo_stack.pop()
            uf = pair_ufs[key]
            uf['parent'][ry] = ry
            uf['rank'][rx] = old_rank
    
    def available_colors(v):
        """Return list of colors that can be assigned to vertex v."""
        avail = []
        for c in range(max_colors):
            # Check adjacency: no neighbor has color c
            if any(color[u] == c for u in adj[v]):
                continue
            # Check bicolored cycle: for each neighbor u with color cu != -1,
            # adding edge (v,c)-(u,cu) must not create a cycle in pair (c,cu)
            ok = True
            for u in adj[v]:
                cu = color[u]
                if cu != -1 and cu != c:
                    if c > cu:
                        c1, c2 = cu, c
                    else:
                        c1, c2 = c, cu
                    key = (c1, c2)
                    if key in pair_ufs:
                        uf = pair_ufs[key]
                        if find(uf, v) == find(uf, u):
                            ok = False
                            break
            if ok:
                avail.append(c)
        return avail
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        # MRV: pick uncolored vertex with fewest available colors
        min_avail = max_colors + 1
        best_v = -1
        for v in range(idx, n):
            if color[v] == -1:
                avail = available_colors(v)
                if len(avail) < min_avail:
                    min_avail = len(avail)
                    best_v = v
                    if min_avail == 0:
                        break
        if best_v == -1:
            return True  # all colored
        if min_avail == 0:
            return False
        
        avail = available_colors(best_v)
        # Shuffle for diversity
        random.shuffle(avail)
        
        for c in avail:
            color[best_v] = c
            # Do unions for all edges from best_v to colored neighbors
            unions_done = []
            valid = True
            for u in adj[best_v]:
                cu = color[u]
                if cu != -1 and cu != c:
                    if do_union(c, cu, best_v, u):
                        unions_done.append(True)
                    else:
                        valid = False
                        break
            if valid:
                # Continue: find next uncolored vertex
                next_idx = best_v + 1
                # Actually, let's just recurse - we'll find the next uncolored
                if backtrack(next_idx):
                    return True
            # Undo
            color[best_v] = -1
            for _ in range(len(unions_done)):
                undo()
        return False
    
    # Order vertices by degree descending for better pruning
    # But MRV handles this dynamically
    
    return backtrack(0)

# EVOLVE-BLOCK-END
