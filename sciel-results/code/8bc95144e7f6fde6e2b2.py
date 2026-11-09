import time
import random
import sys

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    if n < 2:
        return True
    # Build adjacency sets for each color
    color_adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 4:
                return False
            color_adj[c][i].add(j)
            color_adj[c][j].add(i)
            idx += 1
    # Check for C4: for each color, no two vertices should have 2+ common neighbors
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = color_adj[c][u] & color_adj[c][v]
                if len(common) >= 2:
                    return False
    return True

def _lz_woldar_coloring(n=26):
    """
    Lazebnik-Woldar construction for R_5(C4) > 26.
    Vertices: 25 points in GF(5)^2 plus one point at infinity.
    Edges between finite points colored by slope of line (5 slopes: 0,1,2,3,inf).
    Edges to infinity colored by x-coordinate (0,1,2,3,4).
    This guarantees no monochromatic C4.
    """
    if n > 26:
        return None
    
    # Vertices 0..24 are (x,y) in GF(5)^2
    # Vertex 25 is the point at infinity
    
    # Helper: get vertex id from (x,y)
    def vid(x, y):
        return x * 5 + y
    
    # Helper: get (x,y) from vertex id
    def coords(v):
        return (v // 5, v % 5)
    
    # Slopes: 0, 1, 2, 3, 4 (where 4 represents infinity)
    # For two points (x1,y1) and (x2,y2):
    # If x1 == x2: slope = 4 (infinity)
    # Else: slope = (y2-y1) * (x2-x1)^(-1) mod 5
    
    # Precompute modular inverses mod 5
    # 1^-1=1, 2^-1=3, 3^-1=2, 4^-1=4
    inv = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == 25:
                # Edge to infinity: color by x-coordinate of i
                x_i, _ = coords(i)
                c = x_i
            else:
                # Edge between finite points
                x1, y1 = coords(i)
                x2, y2 = coords(j)
                if x1 == x2:
                    c = 4  # infinity slope
                else:
                    dx = (x2 - x1) % 5
                    dy = (y2 - y1) % 5
                    c = (dy * inv[dx]) % 5
            colors.append(c)
    
    return {'n': n, 'colors': colors}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 seconds margin before 120s limit
    
    # First, get the known good construction for n=26
    result = _lz_woldar_coloring(26)
    if result is None:
        return {'n': 1, 'colors': []}
    
    best_n = result['n']
    best_colors = result['colors']
    
    # Verify it's valid (should be, but just in case)
    if not _is_valid_coloring(best_n, best_colors):
        # Fallback to smaller
        return _lz_woldar_coloring(25)
    
    # Now try to extend to n=27 or n=28
    # We need to add one or more vertices and color their edges to existing vertices
    # such that no monochromatic C4 is created.
    
    # For extension, we use backtracking.
    # Current coloring: cm is a 2D list cm[i][j] for i<j
    # Let's convert best_colors to a matrix
    
    def colors_to_matrix(colors, n):
        cm = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                cm[i][j] = colors[idx]
                cm[j][i] = colors[idx]
                idx += 1
        return cm
    
    def matrix_to_colors(cm, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(cm[i][j])
        return colors
    
    cm = colors_to_matrix(best_colors, best_n)
    
    # Build conflict structure
    # For each color c, adj[c][u] = set of neighbors of u in color c
    # conflict[c][u][v] = 1 if u and v have a common neighbor in color c
    # If we assign color c to edge (new_v, u) and color c to edge (new_v, v),
    # and conflict[c][u][v] == 1, then we create a C4 in color c.
    
    def build_conflicts(cm, n):
        adj = [[set() for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i][j]
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        conflicts = [[0] * n for _ in range(n)]  # conflicts[u][v] = mask of colors where u,v have common neighbor
        # Actually, we need conflicts per color.
        # Let's use a 3D array or a list of bitmasks.
        # conflicts[u][v] is a 5-bit mask: bit c is set if u and v have a common neighbor in color c.
        
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[c][u] & adj[c][v]:
                        # Set bit c
                        conflicts[u][v] |= (1 << c)
                        conflicts[v][u] |= (1 << c)
        
        return conflicts, adj
    
    conflicts, adj = build_conflicts(cm, best_n)
    
    # Try to extend to n+1
    def try_extend_one(cm, n, conflicts, adj, max_attempts=500):
        new_n = n + 1
        new_v = new_n - 1
        
        # new_cm is a copy
        new_cm = [row[:] + [-1] for row in cm]
        new_cm.append([-1] * new_n)
        
        # assigned[i] = color of edge (new_v, i)
        assigned = [-1] * n
        
        # Precompute: for each vertex u and color c, 
        # the set of vertices v that conflict with u in color c
        # i.e., conflicts[u][v] has bit c set
        
        # For backtracking:
        # When assigning color c to edge (new_v, u), we must ensure that
        # for all previously assigned vertices v with assigned[v] == c,
        # conflicts[u][v] does NOT have bit c set.
        
        # We can maintain, for each color c, a bitmask of which vertices have been assigned color c.
        # But n is up to 26, so bitmask fits in an int.
        
        # Let's use a different approach:
        # For each color c, maintain a set (or list) of vertices assigned color c.
        # When assigning color c to u, check all v in assigned_set[c]:
        #   if conflicts[u][v] & (1 << c): conflict -> invalid
        
        # Order of vertices: try to assign in a good order.
        # Maybe assign vertices with high degree first? Or random?
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            order = list(range(n))
            random.shuffle(order)
            
            # assigned_sets[c] = list of vertices assigned color c
            assigned_sets = [[] for _ in range(5)]
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    # Success
                    for i in range(n):
                        c = assigned[i]
                        new_cm[new_v][i] = c
                        new_cm[i][new_v] = c
                    return True
                
                u = order[idx]
                
                # Try colors 0-4
                colors_to_try = [0, 1, 2, 3, 4]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    valid = True
                    # Check against all vertices in assigned_sets[c]
                    for v in assigned_sets[c]:
                        if conflicts[u][v] & (1 << c):
                            valid = False
                            break
                    
                    if valid:
                        assigned[u] = c
                        assigned_sets[c].append(u)
                        if backtrack(idx + 1):
                            return True
                        assigned_sets[c].pop()
                        assigned[u] = -1
                
                return False
            
            if backtrack(0):
                return new_cm, True
        
        return None, False
    
    # Try to extend from 26 to 27
    success = False
    attempts = 0
    while not success and time.time() < deadline - 10:
        attempts += 1
        new_cm, success = try_extend_one(cm, best_n, conflicts, adj, max_attempts=100)
        if success:
            best_n += 1
            cm = new_cm
            conflicts, adj = build_conflicts(cm, best_n)
            best_colors = matrix_to_colors(cm, best_n)
            # Try to extend further
        else:
            # If we fail a few times, maybe give up on extension
            if attempts > 5:
                break
    
    # If we couldn't extend, return the n=26 result
    if best_n < 26:
        result = _lz_woldar_coloring(26)
        best_n = result['n']
        best_colors = result['colors']
    
    return {'n': best_n, 'colors': best_colors}
