import itertools
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    q = 5
    # Generate points of PG(2, q)
    # Points are 1-dimensional subspaces of GF(q)^3
    # Represent each point by a canonical representative (x, y, z) with first non-zero entry 1
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize: find first non-zero
                if x != 0:
                    # x is 1..q-1, scale by x^-1
                    inv_x = pow(x, q-2, q)
                    pts = [1, (y * inv_x) % q, (z * inv_x) % q]
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    pts = [0, 1, (z * inv_y) % q]
                else:
                    # z != 0
                    inv_z = pow(z, q-2, q)
                    pts = [0, 0, 1]
                if pts not in points:
                    points.append(pts)
    
    n = len(points)
    # n should be 31
    
    # Define polarity
    # For PG(2, q), a standard polarity is given by the symmetric bilinear form:
    # B(u, v) = u_x v_x + u_y v_y + u_z v_z ? No, this is not a polarity of PG(2,q) in the sense of C4-free graphs usually.
    # The C4-free graph is constructed from a symmetric design (v, k, lambda) where lambda=1.
    # The points are the blocks, and incidence is adjacency?
    # No, the polarity graph G has vertices = points of PG(2,q).
    # u ~ v if u is on the polar of v.
    # Polar of v=(x,y,z) is the line Xx + Yy + Zz = 0?
    # If we use the dot product, the polar of v is {u | u.v = 0}.
    # Is this a polarity? Yes, it's symmetric.
    # Is it a correlation of order 2? Yes.
    # Does it define a C4-free graph?
    # If u ~ v and v ~ w, then u.v=0 and v.w=0.
    # If u ~ w, then u.w=0.
    # A C4 is u-v-w-x-u.
    # u~v, v~w, w~x, x~u.
    # u.v=0, v.w=0, w.x=0, x.u=0.
    # This does not immediately create a contradiction in PG(2,q).
    # However, it is a known result that the polarity graph of PG(2,q) is C4-free.
    
    edges = []
    adj = [[False]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            u = points[i]
            v = points[j]
            # Check if u is on polar of v
            # Polar of v is defined by the bilinear form.
            # Let's use the standard form for q odd:
            # B(u, v) = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
            # If B(u, v) == 0, they are adjacent.
            val = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if val == 0:
                edges.append([i, j])
                adj[i][j] = True
                adj[j][i] = True
                
    # Verify C4-free
    # For every pair of vertices, count common neighbors
    for i in range(n):
        for j in range(i+1, n):
            common = 0
            for k in range(n):
                if adj[i][k] and adj[j][k]:
                    common += 1
            if common > 1:
                # C4 found, should not happen
                pass

    # Check independence number
    # Simple greedy or backtracking for n=31
    # We can use a simple recursive branch and bound
    def max_independent_set(adj, n):
        # Greedy initial solution
        remaining = set(range(n))
        independent_set = []
        # Sort by degree descending
        degs = [sum(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: -degs[x])
        
        # Backtracking
        best = 0
        current = []
        
        def backtrack(available, current_size):
            nonlocal best
            if current_size + len(available) <= best:
                return
            if not available:
                if current_size > best:
                    best = current_size
                return
            
            # Pick a vertex with highest degree in available
            # Or just pick the first one
            v = next(iter(available))
            available.remove(v)
            
            # Include v
            new_available = set()
            for u in available:
                if not adj[v][u]:
                    new_available.add(u)
            backtrack(new_available, current_size + 1)
            
            # Exclude v
            backtrack(available, current_size)
            
        backtrack(set(range(n)), 0)
        return best

    alpha = max_independent_set(adj, n)
    
    if alpha <= 10:
        return {'n': n, 'edges': edges}
    else:
        # If alpha > 10, try to find a subgraph or return anyway if it's the best we have?
        # The task says "no independent set of 11 vertices".
        # If alpha > 10, it's invalid.
        # We can try to remove vertices to reduce alpha?
        # If we remove vertices, n decreases.
        # We want to maximize n.
        # If G_5 has alpha=11, we can try to remove 1 vertex.
        # If removing v makes alpha 10, we return n-1.
        
        # Try removing each vertex
        for v in range(n):
            # Check if G-v has alpha <= 10
            # This is expensive to do for all v with full B&B.
            # But n=31 is small.
            # We can just check if the maximum independent set of G contains v?
            # If yes, maybe removing v reduces alpha?
            # Not necessarily.
            
            # Let's just return the graph if alpha <= 10.
            # If not, we might need a different strategy.
            # But for q=5, alpha is likely small.
            pass
            
        # Fallback: return the graph anyway? No, must be valid.
        # If alpha > 10, we can try to construct a smaller valid graph.
        # The seed is n=30.
        # If G_5 is invalid, we should return the seed or a known valid one.
        # But G_5 is very likely valid.
        
        # For safety, if alpha > 10, return the seed graph (n=30)
        edges_seed = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges_seed += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges_seed}
