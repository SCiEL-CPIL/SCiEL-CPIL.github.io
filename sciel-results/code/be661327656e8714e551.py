import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 20
    num_colors = 20
    
    # Partitions for bipartite structure
    L = list(range(10))
    R = list(range(10, 20))
    
    # Map R vertices to 0..9 for easier indexing in C[i][j]
    # i in 0..9, j in 0..9 represents edge (i, j+10)
    
    def check_bipartite_4cycle(C, i1, j1, i2, j2):
        """
        Check if the 4-cycle formed by (i1, j1), (i1, j2), (i2, j1), (i2, j2) is rainbow.
        C is a 10x10 list of lists.
        """
        c11 = C[i1][j1]
        c12 = C[i1][j2]
        c21 = C[i2][j1]
        c22 = C[i2][j2]
        return len({c11, c12, c21, c22}) == 4

    def try_construct_bipartite(seed):
        """
        Try to find a valid colouring for K_10,10.
        Returns a 10x10 matrix of colours or None.
        """
        rng = random.Random(seed)
        C = [[-1] * 10 for _ in range(10)]
        
        # Order of assignment: row by row
        # For each cell (i, j), pick a colour
        # Check against all previous rows
        # Optimization: We can pre-compute forbidden colours for (i,j) based on existing cells in row i and col j,
        # and the cross-constraints with previous rows.
        
        for i in range(10):
            for j in range(10):
                forbidden = set()
                # Constraint 1: Distinct in row i
                for j2 in range(10):
                    if j2 != j and C[i][j2] != -1:
                        forbidden.add(C[i][j2])
                        
                # Constraint 2: Distinct in col j
                for i2 in range(10):
                    if i2 != i and C[i2][j] != -1:
                        forbidden.add(C[i2][j])
                        
                # Constraint 3 & 4: Cross constraints with previous rows
                # For any i2 < i, the 4-cycle (i, j), (i, j2), (i2, j), (i2, j2) must be rainbow.
                # This implies C[i][j] cannot be C[i2][j] (already in forb) or C[i2][j2] (if j2 != j)
                # AND C[i][j] cannot be such that C[i][j2] == C[i2][j] (violates rainbow)
                # AND C[i][j2] == C[i2][j2] (violates rainbow)
                # Wait, the condition is that the 4 values are distinct.
                # Let x = C[i][j].
                # We know C[i][j2] for j2 != j is already assigned (if we fill row by row, left to right, then j2 < j are assigned, j2 > j are not).
                # Let's assume we fill row i left to right.
                # So for j2 < j, C[i][j2] is known.
                # For j2 > j, C[i][j2] is -1.
                
                # This incremental check is complex if we don't know future values.
                # Better: Backtrack.
                
                pass
                
        # Simple Backtracking
        def backtrack(i, j):
            if i == 10:
                return True
            if j == 10:
                return backtrack(i + 1, 0)
            
            # Try all colours
            colors = list(range(num_colors))
            rng.shuffle(colors)
            
            for c in colors:
                # Check row constraint
                row_ok = True
                for jj in range(10):
                    if jj != j and C[i][jj] == c:
                        row_ok = False
                        break
                if not row_ok:
                    continue
                
                # Check col constraint
                col_ok = True
                for ii in range(10):
                    if ii != i and C[ii][j] == c:
                        col_ok = False
                        break
                if not col_ok:
                    continue
                
                # Check cross constraints with previous rows
                cross_ok = True
                for ii in range(i):
                    # Check 4-cycle (i, j), (i, jj), (ii, j), (ii, jj) for all jj != j
                    # We need C[i][j], C[i][jj], C[ii][j], C[ii][jj] to be distinct.
                    # C[i][j] = c
                    # C[ii][j] is known
                    # C[ii][jj] is known
                    # C[i][jj] is known if jj < j, else unknown.
                    
                    # If jj > j, C[i][jj] is -1. We can't check fully yet.
                    # This suggests we should check constraints only when all 4 are known.
                    # Or, we can enforce that c is not equal to C[ii][jj] for any jj, 
                    # and that C[i][jj] is not equal to C[ii][j] or C[ii][jj].
                    
                    # Let's just check the ones where C[i][jj] is known (jj < j).
                    for jj in range(j):
                        c_ii_jj = C[ii][jj]
                        c_i_jj = C[i][jj]
                        c_ii_j = C[ii][j]
                        
                        # The 4 values are: c, c_i_jj, c_ii_j, c_ii_jj
                        if len({c, c_i_jj, c_ii_j, c_ii_jj}) < 4:
                            cross_ok = False
                            break
                    
                    if not cross_ok:
                        break
                        
                    # Also need to ensure that for future jj > j, it's possible.
                    # This is hard to guarantee. We rely on backtracking.
                    
                if cross_ok:
                    C[i][j] = c
                    if backtrack(i, j + 1):
                        return True
                    C[i][j] = -1
                    
            return False

        if backtrack(0, 0):
            return C
        return None

    best_edges = []
    best_count = 0
    
    # Phase 1: Find a valid K_10,10
    C_bip = None
    for seed in range(1000):
        if time.time() > deadline:
            break
        C_bip = try_construct_bipartite(seed)
        if C_bip:
            break
            
    if C_bip:
        # Construct edges for K_10,10
        edges = []
        for i in range(10):
            for j in range(10):
                u = i
                v = j + 10
                c = C_bip[i][j]
                edges.append([u, v, c])
        best_count = len(edges)
        best_edges = edges
        
        # Phase 2: Add remaining edges
        # Remaining edges: (u, v) where u, v < 10 or u, v >= 10
        remaining_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                # Check if already in edges
                is_bip = (u < 10 and v >= 10) or (u >= 10 and v < 10)
                if not is_bip:
                    remaining_edges.append((u, v))
                    
        # Build adjacency and colour map for quick checking
        adj = [[-1] * n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            
        def is_valid_edge(u, v, c):
            # Check all 4-cycles involving (u, v) with colour c
            # A 4-cycle is u-v-w-x-u.
            # Edges: (u,v), (v,w), (w,x), (x,u).
            # Colours: c, adj[v][w], adj[w][x], adj[x][u].
            # Must be distinct.
            
            # Iterate over all w, x
            # To be efficient, iterate over neighbors of v and u.
            # w must be neighbor of v.
            # x must be neighbor of u.
            # And (w,x) must be an edge.
            
            for w in range(n):
                if w == v or w == u:
                    continue
                if adj[v][w] == -1:
                    continue
                cw = adj[v][w]
                if cw == c:
                    return False
                
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if adj[u][x] == -1 or adj[w][x] == -1:
                        continue
                    cx = adj[u][x]
                    cwx = adj[w][x]
                    
                    if cx == c or cx == cw or cx == cwx:
                        return False
                    if cwx == c or cwx == cw:
                        return False
                        
            return True

        # Try to add remaining edges
        rng = random.Random(42)
        rng.shuffle(remaining_edges)
        
        for u, v in remaining_edges:
            if time.time() > deadline:
                break
                
            # Try to find a valid colour
            colors = list(range(num_colors))
            rng.shuffle(colors)
            added = False
            for c in colors:
                if is_valid_edge(u, v, c):
                    edges.append([u, v, c])
                    adj[u][v] = c
                    adj[v][u] = c
                    best_count += 1
                    added = True
                    break
                    
    else:
        # Fallback to simple greedy if bipartite fails
        # (Unlikely to fail, but just in case)
        pass

    return best_edges
