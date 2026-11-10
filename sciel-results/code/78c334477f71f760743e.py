import time
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Order vertices by degree (all 4, so just index order or random shuffle for symmetry breaking?)
    # Let's stick to index order for determinism, or sort by index.
    # Actually, high degree first is standard, but all are 4.
    # We can try to break symmetry by fixing the color of vertex 0 to 0.
    
    best_colors = list(range(n))
    best_num = n
    
    deadline = time.time() + 55 # 55s safety margin for 60s limit
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline:
            break
        if k >= best_num:
            break
            
        # State
        colors = [-1] * n
        # DSU for each pair of colors
        # pairs[i][j] for i < j
        # We need to store parent and rank for each pair.
        # Since n is small, we can just use lists.
        
        # To speed up, we only maintain DSU for pairs that are currently "active" or we check on the fly?
        # Checking on the fly with full DSU reconstruction is O(E) per step, which is 100 * 50 = 5000 ops.
        # With backtracking, this might be too slow if branching factor is high.
        # Incremental DSU is better.
        
        # Initialize DSUs
        # There are k*(k-1)/2 pairs.
        # Let's create a 2D array of DSUs.
        # dsu[i][j] = (parent, rank) for i < j
        
        dsus = {}
        for i in range(k):
            for j in range(i+1, k):
                dsus[(i,j)] = {'parent': list(range(n)), 'rank': [0]*n}
                
        def find(dsu, x):
            p = dsu['parent']
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x
            
        def union(dsu, x, y):
            p = dsu['parent']
            r = dsu['rank']
            px = find(dsu, x)
            py = find(dsu, y)
            if px == py:
                return False
            if r[px] < r[py]:
                px, py = py, px
            p[py] = px
            if r[px] == r[py]:
                r[px] += 1
            return True
            
        def get_dsu(c1, c2):
            if c1 > c2:
                c1, c2 = c2, c1
            return dsus[(c1, c2)]
            
        # Backtracking
        # Order vertices
        order = list(range(n))
        # Fix vertex 0 to color 0 to break symmetry
        colors[0] = 0
        # No unions needed for vertex 0 as no neighbors colored yet.
        
        # We need to track which vertices are colored to know which neighbors to check?
        # Or just check all neighbors?
        # If neighbor is uncolored, skip.
        
        found = [False]
        
        def backtrack(idx):
            if time.time() > deadline:
                return
            if found[0]:
                return
            if idx == n:
                found[0] = True
                return
                
            v = order[idx]
            
            # Try each color
            for c in range(k):
                # Check if c is valid for v
                valid = True
                # Check neighbors
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                    if colors[nb] == -1:
                        continue
                    # Check if adding edge (v, nb) creates cycle in pair (c, colors[nb])
                    c_nb = colors[nb]
                    if c == c_nb:
                        valid = False
                        break
                    dsu = get_dsu(c, c_nb)
                    if not union(dsu, v, nb):
                        # Cycle created
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # If valid, assign color and record unions to undo
                colors[v] = c
                unions_to_undo = []
                for nb in adj[v]:
                    c_nb = colors[nb]
                    if c_nb == -1 or c_nb == c:
                        continue
                    dsu = get_dsu(c, c_nb)
                    # We already unioned in the check above?
                    # Wait, the check above mutated the DSU.
                    # We need to restore the DSU state if we backtrack.
                    # This is tricky with incremental DSU.
                    # Alternative: Don't union in the check. Just check if connected.
                    # Then if valid, union.
                    
                    # Let's re-implement:
                    # 1. Check adjacency and connectivity without union.
                    # 2. If all good, union and recurse.
                    # 3. Undo unions on return.
                    
                    # My current logic unioned during check. This is bad for backtracking unless we log changes.
                    # Let's change strategy:
                    # In the loop above, I should NOT union. Just check find(v) == find(nb).
                    # If they are different, it's safe.
                    # Then, if we decide to take this branch, we union.
                    
                    # I will rewrite the inner loop to be "check only".
                    
                    pass # Placeholder, logic needs rewrite below
                
                # The logic above is flawed for backtracking because it mutates state during check.
                # I will implement a "check" function that doesn't mutate, and a "commit" function.
                
                # Actually, for k=3, n=50, the search space is small.
                # Maybe just reconstructing the DSU from scratch for each candidate color is fast enough?
                # Reconstructing DSU for a pair takes O(E) = 100 ops.
                # 3 pairs * 100 = 300 ops per color check.
                # 3 colors * 300 = 900 ops per vertex.
                # 50 vertices * 900 = 45,000 ops per path.
                # Backtracking might explore many paths.
                # But with pruning, it should be fine.
                
                # Let's use the "reconstruct DSU for check" approach for simplicity and correctness.
                
                valid = True
                for c_try in range(k):
                    if colors[v] == -1: # We are trying c_try for v
                         pass
                    # Check neighbor conflicts
                    for nb in adj[v]:
                        if colors[nb] == c_try:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check cycle for each pair involving c_try
                    # We need to simulate the coloring: v has c_try, others as in `colors`
                    # For each pair (c_try, c_other), check if edges between these colors form a forest.
                    # We can do this by building a DSU for that specific pair using current `colors` + v=c_try.
                    
                    cycle_found = False
                    for c_other in range(k):
                        if c_other == c_try:
                            continue
                        # Build DSU for pair (c_try, c_other)
                        parent = list(range(n))
                        rank = [0]*n
                        def find_local(x):
                            while parent[x] != x:
                                parent[x] = parent[parent[x]]
                                x = parent[x]
                            return x
                        def union_local(x, y):
                            px, py = find_local(x), find_local(y)
                            if px == py:
                                return False
                            if rank[px] < rank[py]:
                                px, py = py, px
                            parent[py] = px
                            if rank[px] == rank[py]:
                                rank[px] += 1
                            return True
                            
                        # Iterate all edges
                        for u, w in edges:
                            cu = colors[u]
                            cw = colors[w]
                            # If u or w is v, use c_try
                            if u == v:
                                cu = c_try
                            if w == v:
                                cw = c_try
                                
                            if cu == -1 or cw == -1:
                                continue
                                
                            if (cu == c_try and cw == c_other) or (cu == c_other and cw == c_try):
                                if not union_local(u, w):
                                    cycle_found = True
                                    break
                        if cycle_found:
                            break
                            
                    if not cycle_found:
                        # Valid color
                        colors[v] = c_try
                        if backtrack(idx + 1):
                            return True
                        colors[v] = -1
                        
                return False
                
        # The nested function definition inside the loop is inefficient.
        # I will define a cleaner backtracking function outside.
        
        # Reset colors
        colors = [-1] * n
        colors[0] = 0
        
        def bt(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            v = order[idx]
            for c in range(k):
                # 1. Check neighbor color conflict
                ok = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # 2. Check bicolored cycles
                # Simulate v=c
                cycle = False
                for c2 in range(k):
                    if c2 == c:
                        continue
                    parent = list(range(n))
                    rank = [0]*n
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    def union(x, y):
                        px, py = find(x), find(y)
                        if px == py:
                            return False
                        if rank[px] < rank[py]:
                            px, py = py, px
                        parent[py] = px
                        if rank[px] == rank[py]:
                            rank[px] += 1
                        return True
                        
                    for u, w in edges:
                        cu = colors[u]
                        cw = colors[w]
                        if u == v: cu = c
                        if w == v: cw = c
                        if cu == -1 or cw == -1:
                            continue
                        if (cu == c and cw == c2) or (cu == c2 and cw == c):
                            if not union(u, w):
                                cycle = True
                                break
                    if cycle:
                        break
                        
                if not cycle:
                    colors[v] = c
                    if bt(idx + 1):
                        return True
                    colors[v] = -1
            return False
            
        if bt(1): # Start from 1 as 0 is fixed
            best_colors = colors[:]
            best_num = k
            break
            
    return best_colors
