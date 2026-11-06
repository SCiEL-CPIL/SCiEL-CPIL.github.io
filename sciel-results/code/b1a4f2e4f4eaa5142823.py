"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def verify_no_c6(colors, n):
        """Verify no color class contains a C6 using bitset adjacency."""
        for c in range(4):
            adj = [0] * n  # bitmask adjacency
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    idx += 1
            # Check for C6: find 6 distinct vertices v0..v5 with edges
            # v0-v1-v2-v3-v4-v5-v0
            for v0 in range(n):
                n0 = adj[v0]
                if n0 == 0:
                    continue
                for v1 in range(n):
                    if not (n0 >> v1) & 1 or v1 == v0:
                        continue
                    n1 = adj[v1] & ~(1 << v0)
                    if n1 == 0:
                        continue
                    for v2 in range(n):
                        if not (n1 >> v2) & 1 or v2 in (v0, v1):
                            continue
                        n2 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                        if n2 == 0:
                            continue
                        for v3 in range(n):
                            if not (n2 >> v3) & 1 or v3 in (v0, v1, v2):
                                continue
                            n3 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            if n3 == 0:
                                continue
                            for v4 in range(n):
                                if not (n3 >> v4) & 1 or v4 in (v0, v1, v2, v3):
                                    continue
                                n4 = adj[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                                if n4 == 0:
                                    continue
                                for v5 in range(n):
                                    if not (n4 >> v5) & 1 or v5 in (v0, v1, v2, v3, v4):
                                        continue
                                    if (adj[v5] >> v0) & 1:
                                        return False
        return True

    def backtrack_solve(n, max_nodes=5000000):
        """
        Backtracking search for a C6-free 4-coloring of K_n.
        Edges are assigned in lexicographic order.
        Uses incremental C6 detection: after adding each edge, check if it
        completes a C6 in its color class.
        """
        total_edges = n * (n - 1) // 2
        colors = [-1] * total_edges
        
        # For each color, maintain adjacency as bitmasks
        adj = [[0] * n for _ in range(4)]
        
        # For incremental C6 check: when adding edge (u,v) to color c,
        # we need to check if there exists a path u-a-b-d-e-v (or v-e-d-b-a-u)
        # in color c where a,b,d,e are distinct and not u or v.
        # Equivalently: is there a path of length 4 from u to v in G_c \ {edge uv}?
        # We check: exists a in N_c(u)\{v}, b in N_c(a)\{u,v}, 
        #           d in N_c(b)\{u,v,a}, e in N_c(d)\{u,v,a,b} with e in N_c(v)\{u}
        
        def would_create_c6(c, u, v):
            """Check if adding edge (u,v) to color c creates a C6."""
            # Need path u-a-b-d-e-v in existing edges of color c
            # where a,b,d,e distinct from u,v and each other
            au = adj[c][u] & ~(1 << v)
            if au == 0:
                return False
            av = adj[c][v] & ~(1 << u)
            if av == 0:
                return False
            
            # For each a in N(u)\{v}, for each b in N(a)\{u,v},
            # for each d in N(b)\{u,v,a}, check if d is adjacent to some e in N(v)\{u,a,b,d}
            # and e != d, e not in {u,v,a,b}
            # Actually simpler: check if there's a path of length 4 from u to v
            # u-a-b-d-e-v means a in N(u), b in N(a), d in N(b), e in N(d) ∩ N(v)
            # with all distinct
            
            for a in range(n):
                if not (au >> a) & 1:
                    continue
                ab = adj[c][a] & ~(1 << u) & ~(1 << v)
                if ab == 0:
                    continue
                for b in range(n):
                    if not (ab >> b) & 1:
                        continue
                    bd = adj[c][b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                    if bd == 0:
                        continue
                    for d in range(n):
                        if not (bd >> d) & 1:
                            continue
                        # Check if d is adjacent to any e in N(v)\{u,a,b,d}
                        de_to_v = adj[c][d] & av & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << d)
                        if de_to_v:
                            return True
            return False

        node_count = 0
        edge_order = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_order.append((i, j))
        
        def solve(pos):
            nonlocal node_count
            if time.time() > deadline - 3:
                return False
            if pos == total_edges:
                return True
            
            node_count += 1
            if node_count > max_nodes:
                return False
            
            u, v = edge_order[pos]
            
            for c in range(4):
                if would_create_c6(c, u, v):
                    continue
                
                # Assign color c
                colors[pos] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if solve(pos + 1):
                    return True
                
                # Undo
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[pos] = -1
            
            return False

        # Use iterative approach to avoid recursion depth issues
        # But for n<=17, total_edges <= 136, so recursion depth is fine
        sys.setrecursionlimit(10000)
        
        if solve(0):
            return colors
        return None

    # Try to find the best coloring, starting from largest n
    for n_try in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline - 5:
            break
        
        result = backtrack_solve(n_try)
        if result is not None:
            # Verify
            if verify_no_c6(result, n_try):
                return {'n': n_try, 'colors': result}
    
    # Fallback: simple valid coloring for small n
    n = 5
    colors = [0] * (n * (n - 1) // 2)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
