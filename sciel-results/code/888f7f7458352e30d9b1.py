"""Rewritten program for R_4(C6) maximization using constructive backtracking with C6 pruning."""
import sys
sys.setrecursionlimit(10000)

def construct():
    # We search for the largest n such that K_n can be 4-colored with no C6 in any color class.
    # Use constructive backtracking: add vertices one by one, assigning colors to edges to previous vertices.
    # Prune if any color class would contain a C6.
    
    # For efficiency, we'll try to find a valid coloring for n=17 first (known possible),
    # then try n=18, n=19, n=20. If we can't find one within time, return the best we have.
    
    import time
    start_time = time.time()
    TIME_LIMIT = 110  # seconds, leave buffer for evaluation overhead
    
    best_n = 1
    best_colors = None
    
    # Try from n=18 down to n=17, then if needed go up from 9
    # Actually, let's try n=20, 19, 18, 17 in order, stopping at first success
    # But since 17 is known possible, and 18+ may be hard, let's be strategic.
    
    # Strategy: Try n=18, 19, 20 with limited search. If fail, return known n=17 coloring.
    
    # First, let's have a known valid coloring for n=17.
    # We'll generate it via a simple heuristic or use a precomputed one.
    # Actually, let's just run the search for n=17 first to get a valid coloring,
    # then try to extend to larger n.
    
    def find_coloring(n, time_limit):
        """Find a 4-coloring of K_n with no C6 in any color class using backtracking."""
        # colors[i][j] for i < j stores the color of edge (i, j)
        # We build vertex by vertex. When adding vertex v, we assign colors to edges (v, u) for u < v.
        
        # To check if adding a C6, we need to check if there exists a path of length 5 in some color
        # that connects back to form a cycle. This is complex, so we use a simpler check:
        # After assigning all edges for a new vertex, check if any color class contains a C6.
        
        # For efficiency, we'll maintain for each color c, the set of edges of that color.
        # When we add a new vertex v with edges to previous vertices, we check if any new C6 is formed.
        
        # A C6 is formed if there exist vertices v0, v1, v2, v3, v4, v5 (distinct) such that
        # all edges (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) have the same color.
        
        # When we add vertex v = v5 (the newest), the C6 must include v. So we check:
        # For each color c, and for each pair (a, b) where a, b < v, if edge (a,b) has color c,
        # and there exist vertices x, y, z < v such that edges (v,x), (x,y), (y,z), (z,a) all have color c...
        # This is still complex. Let's just do a full check after each vertex addition.
        
        # For n up to 20, a full C6 check on each color class might be feasible if done smartly.
        
        # Let's use a different approach: for each color c, maintain the adjacency matrix.
        # When adding vertex v, for each color c, check if v creates a C6 in color c.
        
        # A C6 containing v in color c: v is connected to some u1 in color c.
        # Then u1 is connected to u2 in color c, u2 to u3, u3 to u4, u4 to u5, u5 to v.
        # All u1..u5 are distinct and not v.
        
        # So for each color c, and for each u1 in neighbors_c(v), we need to check if there's a path of length 4
        # from u1 to some u5 in neighbors_c(v) with all intermediate vertices distinct from v and each other.
        
        # This is still O(n^5) per check in the worst case, which is too slow.
        
        # Alternative: precompute for each color c and each pair (a,b), whether there's a path of length k
        # between a and b in color c. But this changes as we add vertices.
        
        # Let's just do a straightforward backtracking with a C6 check that's optimized.
        
        # For the check: when we've assigned colors for vertices 0..v, we check if any color class has a C6.
        # We can do this by, for each color c, checking all 6-tuples... which is O(n^6), too slow.
        
        # Better: for each color c, build the adjacency list. Then for each pair of edges (a,b) and (c,d) 
        # in color c, check if they can be part of a C6. Still complex.
        
        # Let's use a different strategy: since n is at most 20, and we're doing backtracking,
        # the search space is 4^(n*(n-1)/2) which is huge. We need strong pruning.
        
        # Key insight: when adding vertex v, the only new C6s are those that include v.
        # So we only need to check for C6s containing v.
        
        # A C6 containing v: v - a - b - c - d - e - v, where all edges have the same color.
        # So for each color col:
        #   For each a in neighbors_col(v):
        #     For each b in neighbors_col(a) where b != v:
        #       For each c in neighbors_col(b) where c not in {v, a}:
        #         For each d in neighbors_col(c) where d not in {v, a, b}:
        #           For each e in neighbors_col(d) where e not in {v, a, b, c}:
        #             If e is in neighbors_col(v), then we have a C6.
        
        # This is O(deg(v)^5) in the worst case, which for n=20 could be 20^5 = 3.2M, which is borderline.
        # But with backtracking, we'll have many calls. Let's optimize.
        
        # Actually, let's just implement it and hope it's fast enough for n <= 18.
        
        colors = {}  # (i, j) -> color for i < j
        adj = [[set() for _ in range(n)] for _ in range(4)]  # adj[col][i] = set of neighbors of i in color col
        
        def get_edge(i, j):
            if i > j:
                i, j = j, i
            return colors.get((i, j), -1)
        
        def set_edge(i, j, col):
            if i > j:
                i, j = j, i
            colors[(i, j)] = col
            adj[col][i].add(j)
            adj[col][j].add(i)
        
        def has_c6_with_vertex(v):
            """Check if adding vertex v creates a C6 in any color class."""
            for col in range(4):
                neighbors_v = adj[col][v]
                for a in neighbors_v:
                    for b in adj[col][a]:
                        if b == v or b == a:
                            continue
                        for c in adj[col][b]:
                            if c in (v, a, b):
                                continue
                            for d in adj[col][c]:
                                if d in (v, a, b, c):
                                    continue
                                for e in adj[col][d]:
                                    if e in (v, a, b, c, d):
                                        continue
                                    if e in neighbors_v:
                                        return True
            return False
        
        def backtrack(v):
            if v == n:
                return True
            if time.time() - start_time > time_limit:
                return False
            
            # Try all 4^v combinations for edges from v to 0..v-1
            # Use recursion with early pruning
            def assign_edges(v, u, assignment):
                if u == 0:
                    # All edges assigned, check for C6
                    for col in range(4):
                        for u2 in range(v):
                            col_val = assignment[u2]
                            set_edge(u2, v, col_val)
                    if has_c6_with_vertex(v):
                        for u2 in range(v):
                            col_val = assignment[u2]
                            adj[col_val][u2].discard(v)
                            adj[col_val][v].discard(u2)
                            del colors[(min(u2,v), max(u2,v))]
                        return False
                    return backtrack(v + 1)
                
                if time.time() - start_time > time_limit:
                    return False
                
                for col in range(4):
                    assignment[u] = col
                    if assign_edges(v, u - 1, assignment):
                        return True
                assignment[u] = -1
                return False
            
            assignment = [-1] * v
            return assign_edges(v, v - 1, assignment)
        
        if backtrack(0):
            # Extract colors in lexicographic order
            result = []
            for i in range(n):
                for j in range(i + 1, n):
                    result.append(colors[(i, j)])
            return result
        return None
    
    # Try n=17 first to get a valid coloring
    result17 = find_coloring(17, 30)
    if result17 is not None:
        best_n = 17
        best_colors = result17
    
    # Try n=18
    if time.time() - start_time < TIME_LIMIT - 20:
        result18 = find_coloring(18, TIME_LIMIT - 25)
        if result18 is not None:
            best_n = 18
            best_colors = result18
    
    # Try n=19
    if best_n < 19 and time.time() - start_time < TIME_LIMIT - 15:
        result19 = find_coloring(19, TIME_LIMIT - 20)
        if result19 is not None:
            best_n = 19
            best_colors = result19
    
    # Try n=20
    if best_n < 20 and time.time() - start_time < TIME_LIMIT - 10:
        result20 = find_coloring(20, TIME_LIMIT - 15)
        if result20 is not None:
            best_n = 20
            best_colors = result20
    
    if best_colors is None:
        # Fallback: return a trivial valid coloring for small n
        n = 5
        best_colors = [0] * (n * (n - 1) // 2)
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
